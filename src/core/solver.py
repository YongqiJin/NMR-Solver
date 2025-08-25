# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing.pool
import numpy as np
import multiprocessing
from multiprocessing import Pool
from functools import partial
from typing import Dict, List, Tuple, Optional
import logging
import copy
from rdkit import Chem

from src.faiss_server.search import search_db

from src.utils.nmrnet import predict_nmr_from_mol
from src.utils.chem_tools import get_canonical_smiles_from_mol, remove_stereo
from src.utils.nmr_match import set_match_score_weighted
from src.utils.base_logger import Logger

from src.core.pool import NMRMolPool
from src.core.operation import replace_halogens, check_validity, mol_ok
from src.core.constraint import satisfy_constraints
from src.core.solver_utils import cut_mols_into_frags, vector_encoding, get_nmr_from_fragment, search_fragments, merge_nmr, filter_mol, process_equi_atoms, set_atom_map_num


def crossover(
    p: multiprocessing.pool.Pool, 
    logger: logging.Logger, 
    config: Dict, 
    H_split: Optional[List], 
    H_shifts: Optional[np.ndarray], 
    C_shifts: Optional[np.ndarray], 
    allowed_elements: Optional[List], 
    nmr_mol_pool: NMRMolPool
) -> NMRMolPool:
    """
    Perform crossover operation to generate new molecules.
    """
    logger.info('-'*20 + ' cut mol into fragments ' + '-'*20)
    
    frag_df = cut_mols_into_frags(p, logger, config, nmr_mol_pool, allowed_elements)

    logger.info('-'*20 + ' calculate fragment vector ' + '-'*20)

    input_list_H, input_list_C = [], []
    for fragment, mol_id in zip(frag_df['fragment'], frag_df['mol_id']):
        atoms_shifts = nmr_mol_pool.atoms_shift_list[mol_id] if mol_id != -1 else None
        input_list_H.append((fragment, atoms_shifts, None, 'H'))
        input_list_C.append((fragment, atoms_shifts, nmr_mol_pool.atoms_equi_class_list[mol_id], 'C'))
    
    use_hnmr = H_shifts is not None
    use_cnmr = C_shifts is not None
    H_shifts_frag = p.starmap(get_nmr_from_fragment, input_list_H) if use_hnmr else None
    C_shifts_frag = p.starmap(get_nmr_from_fragment, input_list_C) if use_cnmr else None
    
    # !!! normalize = False
    vec_frags = vector_encoding(H_shifts_frag, C_shifts_frag, normalize=False)
    vec_target = vector_encoding(H_shifts, C_shifts, normalize=False)
    
    logger.info('vec_frags shape: %s', vec_frags.shape)
    logger.info('vec_target shape: %s', vec_target.shape)
    
    logger.info('-'*20 + ' 1st filter: search by index ' + '-'*20)
    
    D, I = search_fragments(vec_target, vec_frags, frag_df, topk=config['topk'])
    
    logger.info('I shape: (%d, %d) -> %s', I.shape[0], I.shape[0], I.shape)
    
    
    logger.info('-'*20 + ' 2ed filter: sort by vec_frags (l2) ' + '-'*20)

    D_flat = D.flatten()
    topk_min_indices_flat = np.argsort(D_flat)[:config['num_filter_pair']]
    
    # sort
    row_indices, col_indices = np.unravel_index(topk_min_indices_flat, D.shape)
    col_indices = I[row_indices, col_indices]

    # deduplicate
    pair_indices = [(min(i, j), max(i, j)) for i, j in zip(row_indices, col_indices)]
    pair_indices = list(set(pair_indices))
    pair_indices = [(i, j) for i, j in pair_indices if i != -1]
    row_indices = [i for i, _ in pair_indices]
    col_indices = [j for _, j in pair_indices]

    logger.info('filter: %d -> %d', len(D_flat), len(topk_min_indices_flat))
    logger.info('deduplicate: %d -> %d', len(topk_min_indices_flat), len(col_indices))
    

    logger.info('-'*20 + ' 3rd filter: sort by match_score (parent nmr) ' + '-'*20)
    m = len(col_indices)

    row_id_list = [row_indices[i] for i in range(m)]
    col_id_list = [col_indices[i] for i in range(m)]
    if use_hnmr:
        input_list = [(H_shifts_frag[row_id], H_shifts_frag[col_id]) for row_id, col_id in zip(row_id_list, col_id_list)]
        H_nmr_merged_list = p.starmap(merge_nmr, input_list)
    if use_cnmr:
        input_list = [(C_shifts_frag[row_id], C_shifts_frag[col_id]) for row_id, col_id in zip(row_id_list, col_id_list)]
        C_nmr_merged_list = p.starmap(merge_nmr, input_list)

    H_match_score = p.map(partial(set_match_score_weighted, y=H_shifts, weight_matrix=None, sigma=config['sigma_h']), H_nmr_merged_list) if use_hnmr else [0] * m
    C_match_score = p.map(partial(set_match_score_weighted, y=C_shifts, weight_matrix=None, sigma=config['sigma_c']), C_nmr_merged_list) if use_cnmr else [0] * m
    match_score = np.array(H_match_score) + np.array(C_match_score)

    # rerank
    sorted_indices = np.argsort(match_score)[::-1]
    row_id_list = np.array(row_id_list)[sorted_indices].tolist()
    col_id_list = np.array(col_id_list)[sorted_indices].tolist()
    match_score = np.array(match_score)[sorted_indices].tolist()

    smiles_list_gen = filter_mol(p, config, row_id_list, col_id_list, frag_df, nmr_mol_pool)

    logger.info('filter: %d -> %d', m, len(smiles_list_gen))
    logger.info('top 10 match score (parent nmr): %s', match_score[:10])
    
    if len(smiles_list_gen) > 0:
        logger.info('-'*20 + ' 4th filter: calculate nmr ' + '-'*20)
        result = predict_nmr_from_mol(smiles_list_gen, raw=True)
    else:
        logger.warning('No valid molecules generated.')
        result = {}
    nmr_mol_gen = NMRMolPool(result)
    nmr_mol_gen.calculate_score(H_split, H_shifts, C_shifts, config, p=p)

    return nmr_mol_gen


def mutate(
    p: multiprocessing.pool.Pool, 
    logger: logging.Logger, 
    config: Dict, 
    H_split: Optional[List], 
    H_shifts: Optional[np.ndarray], 
    C_shifts: Optional[np.ndarray], 
    allowed_elements: Optional[List], 
    nmr_mol_pool: NMRMolPool
) -> NMRMolPool:
    """
    Perform mutation operation to generate new molecules.
    """
    optional_halogens = config['optional_halogens']
    num_mutate_mol = config['num_mutate_mol']
    optional_elements = list(set(allowed_elements) & set(optional_halogens)) if allowed_elements else optional_halogens

    input_list = [(mol, equi_class, optional_elements) for mol, equi_class in zip(nmr_mol_pool.mol_list, nmr_mol_pool.atoms_equi_class_list)]
    results = p.starmap(replace_halogens, input_list)
    mol_list_gen = [item for result in results for item in result]
    smiles_list_gen = p.map(get_canonical_smiles_from_mol, mol_list_gen)
    
    smiles_list_gen = list({smiles for smiles in smiles_list_gen if smiles not in nmr_mol_pool.smiles_list})[:num_mutate_mol]
    logger.info('num_mutate_mol: %d', num_mutate_mol)
    logger.info('optional elements: %s', optional_elements)
    logger.info('num_mol_gen: %d', len(smiles_list_gen))

    if len(smiles_list_gen) > 0:
        logger.info('-'*20 + ' calculate nmr ' + '-'*20)
        result = predict_nmr_from_mol(smiles_list_gen, raw=True)
    else:
        logger.warning('No valid molecules generated.')
        result = {}
    nmr_mol_gen = NMRMolPool(result)
    nmr_mol_gen.calculate_score(H_split, H_shifts, C_shifts, config, p=p)
    
    return nmr_mol_gen


def run_solver(
    config: Dict, 
    H_split: Optional[List] = None, 
    H_shifts: Optional[np.ndarray] = None, 
    C_shifts: Optional[np.ndarray] = None, 
    allowed_elements: Optional[List[str]] = None, 
    constraints: Optional[Dict] = None, 
    candidates: Optional[List[str]] = None, 
    logger: Optional[logging.Logger] = Logger().get_logger(),
) -> Tuple[NMRMolPool, Dict[str, Dict]]:
    """
    Main solver function to optimize molecules based on NMR data.
    """
    # if logger is None, no logging
    if logger is None:
        logger = logging.getLogger("default_logger")
        logger.addHandler(logging.NullHandler())
    
    logger.info('-'*20 + ' target ' + '-'*20)
    
    H_split = H_split if config['use_H_split'] else None
    H_shifts = H_shifts.astype(np.float32) if H_shifts is not None else None
    C_shifts = C_shifts.astype(np.float32) if C_shifts is not None else None
    
    invalid_patterns_list = []
    for pattern in config['invalid_patterns']:
        try:
            mol = Chem.MolFromSmarts(pattern)
        except:
            logger.warning('Invalid bad pattern: %s', pattern)
            continue
        invalid_patterns_list.append(pattern)
    config['invalid_patterns'] = invalid_patterns_list

    candidates_mol = []
    if candidates is not None:
        for smi in candidates:
            try:
                mol = Chem.MolFromSmiles(smi)
            except:
                logger.warning('Invalid Candidate SMILES: %s', smi)
                continue
            if check_validity(mol):
                candidates_mol.append(mol)
    candidates = [get_canonical_smiles_from_mol(mol) for mol in candidates_mol]
    
    logger.info('H_split: %s', H_split)
    logger.info('H_shifts: %s', H_shifts)
    logger.info('C_shifts: %s', C_shifts)
    logger.info('allowed_elements: %s', allowed_elements)
    logger.info('constraints: %s', constraints)
    logger.info('candidates: %s', candidates)
    logger.info('config: %s', config)


    logger.info('-'*20 + ' search from db ' + '-'*20)
    
    result = search_db(H_shifts, C_shifts, config["num_search"])
    nmr_mol_pool = NMRMolPool(result)
    
    with Pool(processes=16) as p:
        if not config["use_stereo"]:
            nmr_mol_pool.mol_list = p.map(remove_stereo, nmr_mol_pool.mol_list)
        
        # remove molecules with None smiles
        nmr_mol_pool.smiles_list = p.map(partial(get_canonical_smiles_from_mol, remove_hs=True), nmr_mol_pool.mol_list)
        id_list = [i for i, smi in enumerate(nmr_mol_pool.smiles_list) if smi != None and Chem.MolFromSmiles(smi) != None]
        nmr_mol_pool.filter_pool(id_list)
        
        logger.info('num_mol_search: %d', len(nmr_mol_pool))
        
        if candidates:
            result = predict_nmr_from_mol(candidates, raw=True)
            nmr_mol_candidate = NMRMolPool(result)
            logger.info('num_candidate: %d', len(nmr_mol_candidate))
            nmr_mol_pool.add_pool(nmr_mol_candidate)
        
        logger.info('num_mol_init: %d', len(nmr_mol_pool))

        logger.info('-'*20 + ' init match score ' + '-'*20)
        nmr_mol_pool.calculate_score(H_split, H_shifts, C_shifts, config, p=p)
        mid_result = {0: {
            'smiles': nmr_mol_pool.smiles_list,
            'score': nmr_mol_pool.score_list,
        }}
        
        if allowed_elements:
            nmr_mol_pool_filtered = copy.deepcopy(nmr_mol_pool)
            filtered_ids = p.map(partial(satisfy_constraints, constraints=constraints), nmr_mol_pool.mol_list)
            filtered_ids = [i for i, valid in enumerate(filtered_ids) if valid]
            nmr_mol_pool_filtered.filter_pool(filtered_ids)
            top10_score_sum = np.sum(nmr_mol_pool_filtered.score_list[:10])
            nmr_mol_pool_filtered.log_topk(logger, topk=10)
        else:
            top10_score_sum = np.sum(nmr_mol_pool.score_list[:10])
            nmr_mol_pool.log_topk(logger, topk=10)
        
        for n_iter in range(1, config["max_iter"]+1):
        
            logger.info('='*30 + f' iter {n_iter} ' + '='*30)
            
            # filter
            if n_iter > 1:
                nmr_mol_pool.filter_pool(config['num_pool'])

            # process_equi_atoms (deprecated)
            # process_equi_atoms(p, nmr_mol_pool)
            
            # label the mol
            mol_list_with_atom_map = p.map(set_atom_map_num, [(mol, i) for i, mol in enumerate(nmr_mol_pool.mol_list)])
            nmr_mol_pool.mol_list = mol_list_with_atom_map

            logger.info('-'*20 + ' crossover ' + '-'*20)
            nmr_mol_crossover = crossover(p, logger, config, H_split, H_shifts, C_shifts, allowed_elements, nmr_mol_pool)
            
            logger.info('-'*20 + ' merge & filter & rerank ' + '-'*20)
            if n_iter == 1 and allowed_elements:
                filtered_ids = p.map(partial(satisfy_constraints, constraints=constraints), nmr_mol_pool.mol_list)
                filtered_ids = [i for i, valid in enumerate(filtered_ids) if valid]
                nmr_mol_pool.filter_pool(filtered_ids)
            nmr_mol_pool.add_pool(nmr_mol_crossover)
            logger.info('num pool: %d', len(nmr_mol_pool))

            logger.info('-'*20 + ' crossover results ' + '-'*20)
            nmr_mol_pool.log_topk(logger, topk=10)
            mid_result[n_iter] = {
                'smiles': nmr_mol_pool.smiles_list,
                'score': nmr_mol_pool.score_list,
            }

            # mutate (deprecated)
            # logger.info('-'*20 + ' mutate ' + '-'*20)
            # nmr_mol_mutate = mutate(p, logger, config, H_split, H_shifts, C_shifts, allowed_elements, nmr_mol_pool)
                
            # logger.info('-'*20 + ' merge & rerank ' + '-'*20)
            # nmr_mol_pool.add_pool(nmr_mol_mutate)

            # logger.info('-'*20 + ' mutate results ' + '-'*20)
            # nmr_mol_pool.log_topk(logger, topk=10)
            # mid_result[f"{n_iter}_mutate"] = {
            #     'smiles': nmr_mol_pool.smiles_list,
            #     'score': nmr_mol_pool.score_list,
            # }
            
            # check stopping condition
            top10_score_sum_new = np.sum(nmr_mol_pool.score_list[:10])
            logger.info('top10_score_sum_old: %f', top10_score_sum)
            logger.info('top10_score_sum_new: %f', top10_score_sum_new)
            if top10_score_sum_new > top10_score_sum:
                top10_score_sum = top10_score_sum_new
            else:
                break
                
        # filter by constraints
        if constraints:
            logger.info('-'*20 + ' filter by constraints ' + '-'*20)
            filtered_ids = p.map(partial(satisfy_constraints, constraints=constraints), nmr_mol_pool.mol_list)
            filtered_ids = [i for i, valid in enumerate(filtered_ids) if valid]
            nmr_mol_pool.filter_pool(filtered_ids)
            logger.info('num: %d', len(nmr_mol_pool))
        
        # filter by validity
        logger.info('-'*20 + ' filter by validity ' + '-'*20)
        filtered_ids = p.map(partial(mol_ok, max_cycle_length=config['max_cycle_length'], invalid_patterns=config['invalid_patterns']), nmr_mol_pool.mol_list)
        filtered_ids = [i for i, valid in enumerate(filtered_ids) if valid]
        nmr_mol_pool.filter_pool(filtered_ids)
        logger.info('num: %d', len(nmr_mol_pool))
        
        # final results
        nmr_mol_pool.filter_pool(config['topk'])
        logger.info('-'*20 + ' final results ' + '-'*20)
        logger.info('num_final: %d', len(nmr_mol_pool))
        
        nmr_mol_pool.log_topk(logger, topk=10)
        mid_result[-1] = {
            'smiles': nmr_mol_pool.smiles_list,
            'score': nmr_mol_pool.score_list,
        }

    return nmr_mol_pool, mid_result
