from typing import Dict, List, Optional, Any
import numpy as np
import logging
import multiprocessing
from functools import partial
from copy import deepcopy

from src.utils.nmrnet import get_nmr_lists, get_active_hs
from src.utils.nmr_match import set_match_score_weighted
from src.utils.nmr_split import predict_h_split, get_weight_matrix


class NMRMolPool(object):
    """
    A class to represent a pool of molecules with NMR data.
    """

    def __init__(self, data: Dict[str, List] = None) -> None:
        """
        Initialize the MolPool with a list of molecules.

        Args:
            data: Dictionary containing molecule data.
        """
        self.smiles_list = data.get('smiles', [])
        self.mol_list = data.get('mol', [])
        self.atoms_shift_list = data.get('atoms_shift', [])
        self.atoms_element_list = data.get('atoms_element', [])
        self.atoms_equi_class_list = data.get('atoms_equi_class', [])
        
        for attr in ['smiles_list', 'mol_list', 'atoms_shift_list', 'atoms_element_list', 'atoms_equi_class_list']:
            assert len(getattr(self, attr)) == len(self), f"Length of {attr} does not match the number of molecules."
        
        self.is_scored = False
        
    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def calculate_score(
        self, 
        H_split: Optional[List[str]] = None, 
        H_shifts: Optional[np.ndarray] = None, 
        C_shifts: Optional[np.ndarray] = None, 
        config: Optional[Dict] = None,
        p: Optional[multiprocessing.pool.Pool] = None,
    ) -> None:
        """
        Calculate scores for molecules in the pool.
        """
        config = config or {'use_H_split': False, 'split_coef': 0.8, 'sigma_h': 1.0, 'sigma_c': 10.0, 'include_active_hs': 'yes'}
        assert config['include_active_hs'] in ['yes', 'no', 'both'], "include_active_hs must be 'yes', 'no', or 'both'."
        
        if config['include_active_hs'] == 'no':
            active_hs_list = p.map(get_active_hs, self.mol_list) if p else [get_active_hs(mol) for mol in self.mol_list]
            remove_active_h_shifts(self.atoms_shift_list, active_hs_list, p=p)
        
        H_shifts_list, C_shifts_list = get_nmr_lists(self.atoms_shift_list, self.atoms_element_list, self.atoms_equi_class_list, sorted=not config['use_H_split'], p=p)

        if H_shifts is None:
            H_match_score_list = [0] * len(H_shifts_list)
        else:
            if config['use_H_split']:
                H_split_list = p.map(predict_h_split, self.mol_list) if p else \
                    [predict_h_split(mol) for mol in self.mol_list]
                if config['include_active_hs'] == 'no':
                    H_split_list = remove_active_h_splits(H_split_list, active_hs_list, self.atoms_element_list, p=p)
                weight_matrix_list = p.map(partial(get_weight_matrix, H_split=H_split, coef=config['split_coef']), H_split_list) if p else \
                    [get_weight_matrix(H_split_pred=H_split_pred, H_split=H_split, coef=config['split_coef']) for H_split_pred in H_split_list]
                H_match_score_list = [set_match_score_weighted(H_shifts, H_nmr, weight_matrix, config['sigma_h']) for H_nmr, weight_matrix in zip(H_shifts_list, weight_matrix_list)]
            else:
                H_match_score_list = [set_match_score_weighted(H_shifts, H_nmr, None, config['sigma_h']) for H_nmr in H_shifts_list]
            
            if config['include_active_hs'] == 'both':
                active_hs_list = p.map(get_active_hs, self.mol_list) if p else [get_active_hs(mol) for mol in self.mol_list]
                atoms_shift_list_new = deepcopy(self.atoms_shift_list)
                for i in range(len(atoms_shift_list_new)):
                    atoms_shift_list_new[i][active_hs_list[i]] = np.nan
                H_shifts_list_new, _ = get_nmr_lists(atoms_shift_list_new, self.atoms_element_list, self.atoms_equi_class_list, sorted=not config['use_H_split'], only_h=True, p=p)
                
                if config['use_H_split']:
                    H_split_list = remove_active_h_splits(H_split_list, active_hs_list, self.atoms_element_list, p=p)
                    weight_matrix_list = p.map(partial(get_weight_matrix, H_split=H_split, coef=config['split_coef']), H_split_list) if p else \
                        [get_weight_matrix(H_split_pred=H_split_pred, H_split=H_split, coef=config['split_coef']) for H_split_pred in H_split_list]
                    H_match_score_list_new = [set_match_score_weighted(H_shifts, H_nmr, weight_matrix, config['sigma_h']) for H_nmr, weight_matrix in zip(H_shifts_list_new, weight_matrix_list)]
                else:
                    H_match_score_list_new = [set_match_score_weighted(H_shifts, H_nmr, None, config['sigma_h']) for H_nmr in H_shifts_list_new]
                
                for i in range(len(H_match_score_list)):
                    if H_match_score_list[i] < H_match_score_list_new[i]:
                        H_match_score_list[i] = H_match_score_list_new[i]
                        self.atoms_shift_list[i] = atoms_shift_list_new[i]
        
        if C_shifts is None:
            C_match_score_list = [0] * len(C_shifts_list)
        else:
            C_match_score_list = [set_match_score_weighted(C_shifts, C_nmr, None, config['sigma_c']) for C_nmr in C_shifts_list]

        self.H_score_list = H_match_score_list
        self.C_score_list = C_match_score_list
        self.score_list = (np.array(H_match_score_list) + np.array(C_match_score_list)).tolist()

        assert len(self.score_list) == len(self), "Length of score_list does not match the number of molecules."
        self.is_scored = True
        self.rerank_by_score()
        
    def filter_pool(self, id_list: List[int] | int) -> None:
        """
        Filter the molecule pool based on given indices.
        """
        if isinstance(id_list, int):
            id_list = list(range(min(id_list, len(self))))
        self.smiles_list = [self.smiles_list[i] for i in id_list]
        self.mol_list = [self.mol_list[i] for i in id_list]
        self.atoms_shift_list = [self.atoms_shift_list[i] for i in id_list]
        self.atoms_element_list = [self.atoms_element_list[i] for i in id_list]
        self.atoms_equi_class_list = [self.atoms_equi_class_list[i] for i in id_list]
        if self.is_scored:
            self.H_score_list = [self.H_score_list[i] for i in id_list]
            self.C_score_list = [self.C_score_list[i] for i in id_list]
            self.score_list = [self.score_list[i] for i in id_list]

    def add_pool(self, pool: "NMRMolPool") -> None:
        """
        Add molecules from another NMRMolPool to this pool.
        """
        assert self.is_scored == pool.is_scored, "Both pools must have the same scoring status."
        self.mol_list.extend(pool.mol_list)
        self.smiles_list.extend(pool.smiles_list)
        self.atoms_shift_list.extend(pool.atoms_shift_list)
        self.atoms_element_list.extend(pool.atoms_element_list)
        self.atoms_equi_class_list.extend(pool.atoms_equi_class_list)
        if self.is_scored:
            self.H_score_list.extend(pool.H_score_list)
            self.C_score_list.extend(pool.C_score_list)
            self.score_list.extend(pool.score_list)
            self.rerank_by_score()
        
    def rerank_by_score(self) -> None:
        """
        Rerank molecules in the pool by their scores.
        """
        assert self.is_scored, "Scores must be calculated before reranking."
        sorted_index = np.argsort(np.array(self.score_list))[::-1]
        self.filter_pool(sorted_index)
        
    def get_topk(self, topk: Optional[int] = None) -> Dict[str, List]:
        """
        Get the top-k molecules based on scores.
        """
        assert self.is_scored, "Scores must be calculated before getting topk."
        topk = topk or len(self)
        return {
            'smiles': self.smiles_list[:topk],
            'mol': self.mol_list[:topk],
            'atoms_shift': self.atoms_shift_list[:topk],
            'atoms_element': self.atoms_element_list[:topk],
            'atoms_equi_class': self.atoms_equi_class_list[:topk],
            'H_score': self.H_score_list[:topk],
            'C_score': self.C_score_list[:topk],
            'score': self.score_list[:topk],
        }
    
    def log_topk(self, logger: logging.Logger, topk: Optional[int] = None) -> None:
        """
        Log the top-k molecules.
        """
        topk = topk or len(self)
        logger.info('top %d scores: %s', topk, self.score_list[:topk])
        logger.info('top %d smiles: %s', topk, self.smiles_list[:topk])
        logger.info('last score: %s', self.score_list[-1] if self.score_list else 0)


    def __getitem__(self, idx: int | slice) -> Dict[str, Any]:
        """
        Get item(s) from the pool by index or slice.
        """
        if isinstance(idx, slice):
            # process slice
            new_pool = NMRMolPool({
                'smiles': self.smiles_list[idx],
                'mol': self.mol_list[idx],
                'atoms_shift': self.atoms_shift_list[idx],
                'atoms_element': self.atoms_element_list[idx],
                'atoms_equi_class': self.atoms_equi_class_list[idx]
            })
            if self.is_scored:
                new_pool.H_score_list = self.H_score_list[idx]
                new_pool.C_score_list = self.C_score_list[idx]
                new_pool.score_list = self.score_list[idx]
                new_pool.is_scored = True
            return new_pool
        else:
            # process single index
            if idx < 0:
                idx += len(self)
            if not 0 <= idx < len(self):
                raise IndexError(f"Index {idx} out of range for pool size {len(self)}")
            
            data  = {
                'smiles': self.smiles_list[idx],
                'mol': self.mol_list[idx],
                'atoms_shift': self.atoms_shift_list[idx].tolist(),
                'atoms_element': self.atoms_element_list[idx].tolist(),
                'atoms_equi_class': self.atoms_equi_class_list[idx].tolist()
            }
            if self.is_scored:
                data.update({
                    'H_score': self.H_score_list[idx],
                    'C_score': self.C_score_list[idx],
                    'score': self.score_list[idx]
                })
            return data
        
        
def process_shift(args):
    atoms_shift, active_hs = args
    atoms_shift[active_hs] = np.nan
    return atoms_shift

def remove_active_h_shifts(
    atoms_shift_list: List[np.ndarray],
    active_hs_list: List[List[int]],
    p: Optional[multiprocessing.pool.Pool] = None
) -> List[np.ndarray]:
    
    if p:
        return p.map(process_shift, [(atoms_shift_list[i], active_hs_list[i]) for i in range(len(atoms_shift_list))])
    else:
        return [process_shift((atoms_shift_list[i], active_hs_list[i])) for i in range(len(atoms_shift_list))]

def process_split(args):
    H_split, active_hs, atoms_element = args
    active_h_indices = np.where(atoms_element == 1)[0].tolist()
    active_hs_in_split = [active_h_indices.index(h) for h in active_hs if h in active_h_indices]
    H_split = [split for j, split in enumerate(H_split) if j not in active_hs_in_split]
    return H_split

def remove_active_h_splits(
    H_split_list: List[List[Any]], 
    active_hs_list: List[List[int]], 
    atoms_element_list: List[np.ndarray],
    p: Optional[multiprocessing.pool.Pool] = None
) -> List[List[Any]]:
    
    if p:
        return p.map(process_split, [(H_split_list[i], active_hs_list[i], atoms_element_list[i]) for i in range(len(H_split_list))])
    else:
        return [process_split((H_split_list[i], active_hs_list[i], atoms_element_list[i])) for i in range(len(H_split_list))]
    