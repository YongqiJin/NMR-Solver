# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict, Optional, Any
import numpy as np
import os
import yaml
from rdkit.Chem import GetPeriodicTable

from src.utils.data_tools import query_lmdb
from src.utils.chem_tools import not_charged

from src.faiss_server.faiss_client import FaissClient
from src.core.pool import NMRMolPool
from src.core.solver_utils import vector_encoding

# load default configuration from config.yaml
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.yaml")
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)
DEFAULT_LMDB_PATH = config["lmdb_path"]


def search_db(
    H_shifts: Optional[np.ndarray] = None, 
    C_shifts: Optional[np.ndarray] = None, 
    k: int = 1000, 
    lmdb_path: str = DEFAULT_LMDB_PATH,
) -> Dict[str, Any]:
    """
    Search the database using chemical shift vectors.
    
    Returns:
        Dict[str, Any]: A dictionary containing the search results including smiles, mol objects, atoms_shift, atoms_element, and atoms_equi_class.
    """
    vec = vector_encoding(H_shifts=H_shifts, C_shifts=C_shifts, normalize=True, padding=True)

    faiss_client = FaissClient()
    _, I = faiss_client.search(index_name='HC', vectors=vec, k=k)
    mol_ids = I[0].tolist()

    meta_data = query_lmdb(path=lmdb_path, keys=[str(i) for i in mol_ids])
    filter_ids = [i for i in range(len(meta_data)) if not_charged(meta_data[i]['mol'])]
    meta_data = [meta_data[i] for i in filter_ids]
    
    return {
        "smiles": [m['canonical_smiles'] for m in meta_data],
        "mol": [m['mol'] for m in meta_data],
        'atoms_shift': [m['nmr_predict'].astype(np.float32) for m in meta_data],
        'atoms_element': [m['atom_index'] for m in meta_data],
        'atoms_equi_class': [m['equi_class'] for m in meta_data],
    }


def search_filter_rank(
    H_split: Optional[List[str]] = None, 
    H_shifts: Optional[np.ndarray] = None, 
    C_shifts: Optional[np.ndarray] = None, 
    num_search: int = 1000, 
    topk: int = 10, 
    allowed_elements: Optional[List[str]] = None, 
    config: Optional[Dict[str, Any]] = None,
) -> NMRMolPool:
    """
    Search, filter, and rank molecules based on chemical shifts and other criteria.

    Returns:
        NMRMolPool: Filtered and ranked pool of molecules.
    """
    assert H_shifts is not None or C_shifts is not None, "At least one of H_shifts and C_shifts should be provided."
    
    meta_data = search_db(H_shifts, C_shifts, k=num_search)
    nmr_mol_pool = NMRMolPool(data=meta_data)
    
    if allowed_elements is not None:
        pt = GetPeriodicTable()
        allowed_elements = [pt.GetAtomicNumber(elem) for elem in allowed_elements]
        filtered_ids = [i for i, atoms_element in enumerate(nmr_mol_pool.atoms_element_list) if set(atoms_element).issubset(set(allowed_elements))]
        nmr_mol_pool.filter_pool(filtered_ids)
        
    nmr_mol_pool.calculate_score(H_split=H_split, H_shifts=H_shifts, C_shifts=C_shifts, config=config)
    nmr_mol_pool.filter_pool(topk)

    return nmr_mol_pool
