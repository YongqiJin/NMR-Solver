# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from rdkit import Chem
from rdkit.Chem import rdmolfiles
from itertools import product
from collections import defaultdict
import numpy as np
from typing import List


def merge_split(combo: List) -> str:
    """
    Merge splitting patterns.

    Args:
        combo (list): List of splitting patterns.

    Returns:
        str: Merged splitting pattern.
    """
    if 'm' in combo:
        return 'm'
    else:
        if set(combo) == set() or set(combo) == set('s'):
            return 's'
        else:
            combo = [item for item in combo if item != 's']
            return ''.join(sorted(combo))


def predict_split(neighbors: List) -> set:
    """
    Predict splitting patterns based on neighbors.

    Args:
        neighbors (list): List of neighboring atoms.

    Returns:
        set: Set of possible splitting patterns.
    """
    split = ['s','d','t','q','p','h','hept']
    options_list = []
    for item in neighbors:
        options = []
        element, num_h = item
        if num_h < len(split):
            options.append(split[num_h])
        else:
            options.append('m')
        if element in ['O', 'N', 'S']:
            options.append('s')
        options_list.append(options)

    all_combinations = list(product(*options_list))
    split = [merge_split(combo) for combo in all_combinations]
    return set(split)


def predict_h_split(mol: Chem.Mol) -> List:
    """
    Predict splitting patterns for hydrogen atoms in a molecule.

    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object.

    Returns:
        list: List of splitting patterns for hydrogen atoms.
    """
    equi_class = rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)
    
    num_hs_dict = {}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'H':
            idx = atom.GetIdx()
            num_hs = sum([1 for n in atom.GetNeighbors() if n.GetSymbol() == 'H'])
            num_hs_dict[idx] = num_hs
    
    h_split = []
    cache = {}
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'H':
            continue
        idx = atom.GetIdx()
        parents = [n for n in atom.GetNeighbors()]
        assert len(parents) == 1
        parent = parents[0]
        assert parent.GetSymbol() != 'H'
        
        parent_idx = parent.GetIdx()
        if parent_idx not in cache:
            parent_neighbors = [n for n in parent.GetNeighbors() if n.GetSymbol() != 'H']
            parent_neighbors_equi_class = defaultdict(list)
            for n in parent_neighbors:
                parent_neighbors_equi_class[equi_class[n.GetIdx()]].append(n)
            
            parent_neighbors = []
            for n_list in parent_neighbors_equi_class.values():
                num_hs = sum([num_hs_dict[n.GetIdx()] for n in n_list if n.GetSymbol() != 'H'])
                if num_hs > 0:
                    parent_neighbors.append((n_list[0].GetSymbol(), num_hs))

            split = predict_split(parent_neighbors)
            if parent.GetSymbol() in ['O', 'N', 'S']:
                split.add('s')
                
            cache[parent_idx] = split
            
        h_split.append(cache[parent_idx])
    
    return h_split


def get_weight_matrix(H_split_pred: List, H_split: List, coef: float = 0.8) -> np.ndarray:
    """
    Compute weight matrix for splitting pattern matching.

    Args:
        H_split_pred (list): Predicted splitting patterns.
        H_split (list): Target splitting patterns.
        coef (float): Coefficient for mismatch penalty.

    Returns:
        np.ndarray: Weight matrix.
    """
    m = len(H_split_pred)
    result = []
    for mode in H_split:
        if mode in ['m','br']:
            result.append([1]*m)
        else:
            result.append([1 if mode in H_split_pred[i] else coef for i in range(m)])
    return np.array(result)
        

if __name__ == '__main__':
    # Test
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    # draw_mol(mol)
    h_split = predict_h_split(mol)
    h_split = ['m','s','d','t','q','p','h']
    print(h_split)
    print(get_weight_matrix(h_split, h_split))
    