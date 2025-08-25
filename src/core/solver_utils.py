# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing.pool
import numpy as np
from typing import List, Dict, Tuple, Optional
import faiss
import torch
from collections import defaultdict
import pandas as pd
from functools import partial
import os
from rdkit import Chem
from itertools import product
import logging
import multiprocessing

from src.utils.nmr_match import set2vec
from src.utils.chem_tools import get_canonical_smiles_from_mol, not_charged, get_elements_from_mol

from src.core.operation import get_complement_cut_type, cut_non_ring, cut_ring, stitch_fragment_to_smiles, replace_halogens
from src.core.pool import NMRMolPool


def vector_encoding(
    H_shifts: Optional[np.ndarray] = None, 
    C_shifts: Optional[np.ndarray] = None, 
    dim: int = 128, 
    normalize: bool = True,
    padding: bool = False,
) -> np.ndarray:
    """
    Generate target vector based on H_shifts and C_shifts.
    """
    if H_shifts is None and C_shifts is None:
        raise ValueError("Both H_shifts and C_shifts are None. Cannot generate the encoding vector.")
    inputs = H_shifts if H_shifts is not None else C_shifts

    dim_pad = dim if padding else 0
    if isinstance(inputs, np.ndarray):
        H_vec = set2vec(H_shifts, nmr_type='H', dim=dim, normalize=normalize) if H_shifts is not None else np.zeros((1, dim_pad))
        C_vec = set2vec(C_shifts, nmr_type='C', dim=dim, normalize=normalize) if C_shifts is not None else np.zeros((1, dim_pad))
    elif isinstance(inputs, List):
        H_vec = set2vec(H_shifts, nmr_type='H', normalize=normalize) if H_shifts is not None else np.zeros((len(inputs), dim_pad))
        C_vec = set2vec(C_shifts, nmr_type='C', normalize=normalize) if C_shifts is not None else np.zeros((len(inputs), dim_pad))
    
    vec = np.concatenate([H_vec, C_vec], axis=1)
    return vec


def cut_mols_into_frags(
    p: multiprocessing.pool.Pool, 
    logger: logging.Logger, 
    config: Dict, 
    nmr_mol_pool: NMRMolPool, 
    allowed_elements: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Cut molecules into fragments.
    """
    optional_halogens = config['optional_halogens']
    if allowed_elements:
        optional_halogens = list(set(optional_halogens) & set(allowed_elements))
    
    fragment_list_non_ring = [frag for fragments in p.map(cut_non_ring, nmr_mol_pool.mol_list) for frag in fragments]
    fragment_list_ring = [frag for fragments in p.map(cut_ring, nmr_mol_pool.mol_list) for frag in fragments]
    fragment_cut_type_list = fragment_list_non_ring + fragment_list_ring
    frag_list, cut_type_list = zip(*fragment_cut_type_list)

    logger.info('non_ring fragments: %d', len(fragment_list_non_ring))
    logger.info('ring fragments: %d', len(fragment_list_ring))
    logger.info('total fragments: %d', len(fragment_cut_type_list))

    smi_list = p.map(get_canonical_smiles_from_mol, frag_list)
    
    frag_df = pd.DataFrame({
        'fragment': frag_list,
        'cut_type': cut_type_list,
        'smiles': smi_list
    })
    # deduplicate by smiles
    frag_df.drop_duplicates(subset=['smiles', 'cut_type'], inplace=True)
    logger.info('unique fragments: %d', len(frag_df))
    
    # filter by allowed elements & charged
    frag_df['elements'] = p.map(get_elements_from_mol, frag_df['fragment'])
    if allowed_elements:
        elements_filter = set(allowed_elements) | {'*'}
        if len(set(config['optional_halogens']) & set(allowed_elements)) > 0:
            elements_filter = set(elements_filter) | set(config['optional_halogens'])
        frag_df = frag_df[frag_df['elements'].apply(lambda x: set(x).issubset(elements_filter))]
    
    frag_df = frag_df[p.map(not_charged, frag_df['fragment'])]
    logger.info('filtered fragments: %d', len(frag_df))
    
    frag_df['mol_id'] = frag_df['fragment'].apply(lambda x: x.GetAtoms()[0].GetAtomMapNum()//1000)
    
    # # replace halogens (old version -- deprecated)
    # if optional_halogens and config['num_mutate_mol'] > 0:
    #     frag_df_mutate = frag_df[frag_df['mol_id'] < config['num_mutate_mol']]
    #     frag_df_mutate['elements_halogen'] = frag_df_mutate['elements'].apply(lambda x: set(x) & set(config['optional_halogens']))
    #     frag_df_mutate = frag_df_mutate[frag_df_mutate['elements_halogen'].apply(lambda x: len(x) > 0)]
    #     logger.info('mutated fragments: %d', len(frag_df_mutate))
    #     results = p.map(partial(replace_halogens, optional_elements=optional_halogens), frag_df_mutate['fragment'])
    #     frag_list = []
    #     cut_type_list = []
    #     mol_id_list = []
    #     for i, result in enumerate(results):
    #         frag_list += result
    #         cut_type_list += [frag_df_mutate['cut_type'].iloc[i]] * len(result)
    #         mol_id_list += [frag_df_mutate['mol_id'].iloc[i]] * len(result)
    #     smi_list = p.map(get_canonical_smiles_from_mol, frag_list)
    #     frag_df_new = pd.DataFrame({
    #         'fragment': frag_list,
    #         'cut_type': cut_type_list,
    #         'smiles': smi_list,
    #         'mol_id': mol_id_list,
    #         'elements': [get_elements_from_mol(frag) for frag in frag_list]
    #     })
    #     frag_df = pd.concat([frag_df, frag_df_new], axis=0)
    #     if allowed_elements:
    #         frag_df = frag_df[frag_df['elements'].apply(lambda x: set(x).issubset(allowed_elements + ['*']))]
        
    #     frag_df.drop_duplicates(subset=['smiles', 'cut_type'], inplace=True)
    #     logger.info('after mutation: %d', len(frag_df))

    frag_df['mol_id'] = frag_df['mol_id'].astype(int)
    
    # add X
    if optional_halogens:
        smiles_X = [f"[1*][{atom}]" for atom in optional_halogens]
        fragment_X = [Chem.MolFromSmarts(smi) for smi in smiles_X]
        cut_type_X = [((atom, 'C'), '-') for atom in optional_halogens]
        elements_X = [[atom, '*'] for atom in optional_halogens]
        mol_id_X = [-1] * len(smiles_X)
        frag_df_X = pd.DataFrame({'fragment': fragment_X, 'cut_type': cut_type_X, 'smiles': smiles_X, 
                                'elements': elements_X, 'mol_id': mol_id_X})
        frag_df = pd.concat([frag_df, frag_df_X], axis=0)
    
    logger.info('final fragments: %d', len(frag_df))

    frag_df.sort_values(by='cut_type', inplace=True, kind='mergesort')
    frag_df['frag_id'] = range(len(frag_df))
    
    logger.info('example fragments: \n%s', frag_df.head(5).to_string(index=False))

    return frag_df
    
    
def get_nmr_from_fragment(
    fragment: Chem.Mol, 
    atoms_shift: Optional[List[float]] = None, 
    atoms_equi_class: Optional[List[int]] = None,
    nmr_type: str = 'H'
) -> np.ndarray:
    """
    Extract NMR data from a fragment.
    """
    if atoms_shift is None:
        return np.array([])
    
    if nmr_type == 'H':
        h_atoms = [atom for atom in fragment.GetAtoms() if atom.GetSymbol() == 'H']
        result = np.array([atoms_shift[atom.GetAtomMapNum()%1000] for atom in h_atoms])
        result = result[~np.isnan(result)]
        return np.sort(result)
    else:
        if atoms_equi_class is None:
            raise ValueError("atoms_equi_class cannot be None for 'C' NMR type.")
        c_atoms = [atom for atom in fragment.GetAtoms() if atom.GetSymbol() == 'C']
        equi_class_dict = {}
        for atom in c_atoms:
            idx = atom.GetAtomMapNum() % 1000
            equi_class_dict[atoms_equi_class[idx]] = atoms_shift[idx]

        return np.sort(list(equi_class_dict.values()))


def build_index(vecs: np.ndarray, M: int = 16, efConstruction: int = 40) -> faiss.IndexHNSWFlat:
    """
    Build a FAISS index for fast vector search.
    """
    index = faiss.IndexHNSWFlat(vecs.shape[1], M)
    index.hnsw.efConstruction = efConstruction
    index.add(vecs)
    return index


def search_fragments(vec_target: np.ndarray, vec_frags: np.ndarray, frag_df: pd.DataFrame, topk: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for matching fragments based on vector similarity.
    """    
    frag_ids_dict = defaultdict(list)
    for cut_type, group in frag_df.groupby('cut_type'):
        frag_ids_dict[cut_type] = group['frag_id'].tolist()

    def map_query(cut_type):
        atoms, bonds = cut_type
        mapped_atoms = tuple(atoms[i + 1] if atoms[i] == 'C' else '*' for i in range(0, len(atoms), 2))
        return [(mapped_atoms, bonds)]
    
    def map_key(cut_type):
        atoms, bonds = cut_type
        mapped_atoms = tuple(atoms[i] for i in range(0, len(atoms), 2))
        combinations = [
            tuple(atom if mask[i] == 0 else '*' for i, atom in enumerate(mapped_atoms))
            for mask in product([0, 1], repeat=len(mapped_atoms))
        ]
        return [(atoms, bonds) for atoms in combinations]
        
    def get_query_key_clusters(cut_type_list) -> List[Tuple[List, List]]:
        query_cluster = defaultdict(list)
        key_cluster = defaultdict(list)
        for cut_type in cut_type_list:
            for k in map_query(cut_type):
                query_cluster[k].append(cut_type)
            for k in map_key(cut_type):
                key_cluster[k].append(cut_type)
        cluster = []
        for k in query_cluster:
            cluster.append((query_cluster[k], key_cluster[k]))
        return cluster
    
    cut_type_list = list(frag_ids_dict.keys())
    query_key_clusters = get_query_key_clusters(cut_type_list)
    # query_key_clusters = [([cut_type], [get_complement_cut_type(cut_type)]) for cut_type in cut_type_list]
    
    all_I = np.full((len(frag_df), topk), -1, dtype=np.int64)
    all_D = np.zeros((len(frag_df), topk))
    
    for querys, keys in query_key_clusters:
        query_ids, key_ids = [], []
        for ct in querys:
            query_ids += frag_ids_dict[ct]
        for ct in keys:
            key_ids += frag_ids_dict[ct]
        
        use_faiss = True if len(key_ids) > 10000 else False
        if use_faiss:
            index = build_index(vec_frags[key_ids])
            index.hnsw.efSearch = topk
            D, I = index.search(vec_target-vec_frags[query_ids], k=topk)
        else:
            query_vecs = vec_target - vec_frags[query_ids]
            key_vecs = vec_frags[key_ids]
            query_tensor = torch.from_numpy(query_vecs).float().cuda()
            key_tensor = torch.from_numpy(key_vecs).float().cuda()
            
            query_norm = torch.sum(query_tensor**2, dim=1, keepdim=True)
            key_norm = torch.sum(key_tensor**2, dim=1, keepdim=True).t()
            dot_product = torch.mm(query_tensor, key_tensor.t())
            distances = query_norm + key_norm - 2 * dot_product
            
            k = min(topk, len(key_ids))
            topk_values, topk_ids = torch.topk(distances, k=k, dim=1, largest=False)
            
            D = topk_values.cpu().numpy()
            I = topk_ids.cpu().numpy()
            
            if k < topk:
                D_padded = np.full((len(query_ids), topk), float('inf'), dtype=D.dtype)
                I_padded = np.full((len(query_ids), topk), -1, dtype=I.dtype)
                D_padded[:, :k] = D
                I_padded[:, :k] = I
                D, I = D_padded, I_padded
        
        I[I!=-1] = np.array(key_ids)[I[I!=-1]]
        all_I[query_ids, :] = I
        all_D[query_ids, :] = D
    
    return all_D, all_I


def merge_nmr(nmr1: np.ndarray, nmr2: np.ndarray) -> np.ndarray:
    """
    Merge two NMR datasets.
    """
    return np.sort(np.concatenate([nmr1, nmr2], axis=0)).astype(np.float32)


def filter_mol(p: multiprocessing.pool.Pool, config: Dict, row_id_list: List[int], col_id_list: List[int], frag_df: pd.DataFrame, nmr_mol_pool: NMRMolPool) -> List[str]:
    """
    Filter molecules based on fragments.
    """
    num_filter_mol = config['num_filter_mol']
    fragment_list = frag_df['fragment'].tolist()
    bond_type_list = [item[-1] for item in frag_df['cut_type']]
    count = 0
    pos = 0
    smiles_list_gen = []
    
    while count < num_filter_mol and pos < len(row_id_list):
        batch_size = min(max(2*(num_filter_mol - count), os.cpu_count()), len(row_id_list) - pos)
        batch_data = [(fragment_list[row_id_list[i]], fragment_list[col_id_list[i]], bond_type_list[row_id_list[i]]) for i in range(pos, pos+batch_size)]
        batch_results = p.map(partial(stitch_fragment_to_smiles, max_cycle_length=config['max_cycle_length'], invalid_patterns=config['invalid_patterns']), batch_data)
        pos += batch_size
        
        for smiles in batch_results:
            if smiles and smiles not in smiles_list_gen and smiles not in nmr_mol_pool.smiles_list:
                smiles_list_gen.append(smiles)
                count += 1
                if count >= num_filter_mol:
                    break
        return smiles_list_gen
    

def classify_equi_atoms(equi_class: List[int], atoms_element: List[int], element_id: int = 6) -> Dict[int, List[int]]:
    """
    Classify equivalent atoms based on their equivalence class.
    """
    equi_dict = defaultdict(list)
    for i, eq in enumerate(equi_class):
        if atoms_element[i] == element_id:
            equi_dict[eq].append(i)
    return dict(equi_dict)


def process_equi_atoms(p: multiprocessing.pool.Pool, nmr_mol_pool: NMRMolPool) -> None:
    """
    Process equivalent atoms in the molecule pool.
    """
    input_list = zip(nmr_mol_pool.atoms_equi_class_list, nmr_mol_pool.atoms_element_list, [6]*len(nmr_mol_pool))
    C_equi_dict_list = p.starmap(classify_equi_atoms, input_list)
    
    for mol_id in range(len(nmr_mol_pool)):
        atoms_shift = nmr_mol_pool.atoms_shift_list[mol_id]
        atoms_element = nmr_mol_pool.atoms_element_list[mol_id]
        C_equi_dict = C_equi_dict_list[mol_id]
        for eq, indices in C_equi_dict.items():
            if len(indices) > 1 and atoms_element[indices[0]] == 6:
                value = np.mean([atoms_shift[i] for i in indices])
                for idx in indices:
                    atoms_shift[idx] = value
        nmr_mol_pool.atoms_shift_list[mol_id] = atoms_shift


def set_atom_map_num(args):
    mol, i = args
    for j, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i * 1000 + j)
    return mol
