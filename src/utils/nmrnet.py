# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import rdmolfiles
import yaml
from unimol_tools import MolPredict

from unimol_tools.utils import logger
import logging
logger.setLevel(logging.ERROR)

with open('./src/utils/nmrnet.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
base_model = config['base_model']
n_splits = config['n_splits']
clf = MolPredict(base_model, n_splits)


def random_split(dataset, random_ratio=0.8):
    if random_ratio == None:
        return dataset, dataset
    else:
        key1 = list(dataset.keys())[0]
        list_length = len(dataset[key1])
        indices = list(range(list_length))
        random.shuffle(indices)
        train_split_index = int(random_ratio * list_length)
        random_set1 = {}
        random_set2 = {}
        for key in dataset:
            random_set1[key] = [dataset[key][i] for i in indices[:train_split_index]]
            random_set2[key] = [dataset[key][i] for i in indices[train_split_index:]]

        return random_set1, random_set2

def filter_data(molecules_nmrdata, filter_list, datatype, sampling_ratio=None):
    filtered_indices = []
    # print("sampling_ratio1", sampling_ratio)
    if 'all' in filter_list:
        if sampling_ratio==None:
            return molecules_nmrdata
        else:
            molecules_nmrdata, molecules_nmrdata2 = random_split(molecules_nmrdata, sampling_ratio)
            return molecules_nmrdata

    for i, atom_mask in enumerate(molecules_nmrdata['atom_mask']):
        if all(x == 0 for x in atom_mask):
            continue
        if any(x in filter_list for x in atom_mask):
            filtered_indices.append(i)
            
    filtered_dict = {}
    for key in molecules_nmrdata.keys():
        filtered_dict[key] = [molecules_nmrdata[key][i] for i in filtered_indices]

    filtered_dict, filtered_dict2 = random_split(filtered_dict, sampling_ratio)

    return filtered_dict

def map_element(nmrtype):
    result = []
    if nmrtype == 'ALL':
        result.append('all')
        return result
    for element in nmrtype.split("+"):
        result.append(Chem.GetPeriodicTable().GetAtomicNumber(element))

    return result

def get_atomic_numbers(mol):
    atomic_numbers = []  
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        atomic_numbers.append(atomic_number)
    return atomic_numbers

def predict_nmr_from_nmrnet(mol_list, clf=clf, merge=False):
    infer_data = {
        'mol': [],
        'atom_target': [],
        'atom_mask': [],
    }
    for i, mol in enumerate(mol_list):

        infer_data['mol'].append(mol)
        infer_data['atom_target'].append([0.0]*512)
        atom_num = get_atomic_numbers(mol)
        infer_data['atom_mask'].append([0]+atom_num+[0]*(512-1-len(atom_num)))

    nmrtype = 'ALL'
    datatype='mol'
    filter_list = map_element(nmrtype)

    filtered_data = infer_data
    filtered_data = filter_data(filtered_data, filter_list, datatype)
    
    # model prediction
    test_pred = clf.predict(filtered_data, datatype = datatype)
    target = clf.cv_true[clf.cv_label_mask]
    predict = clf.cv_pred[clf.cv_label_mask]
    index_mask = np.array(clf.cv_index_mask.tolist()).astype(np.int8)
    data_dict = {
        # 'cv_true': clf.cv_true,
        'cv_pred': clf.cv_pred,
        # 'cv_pred_fold': clf.cv_pred_fold,
        'cv_label_mask': clf.cv_label_mask,
        'index_mask': index_mask
    }
    if not merge:
        return data_dict
    
    nmr_list = []
    index_list = []
    for i in range(len(data_dict['index_mask'])):
        cv_pred=data_dict['cv_pred'][i].astype(np.float32)
        cv_label_mask=data_dict['cv_label_mask'][i]
        index_mask=data_dict['index_mask'][i]
        nmr_predict=cv_pred[cv_label_mask]
        mol_index=index_mask[cv_label_mask]
        nmr_list.append(nmr_predict)
        index_list.append(mol_index)
    return nmr_list, index_list

def get_equi_class(mol):
    equi_class = rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)
    return np.array(equi_class).astype(np.int16)

def merge_equi_nmr(nmr, atom_index, equi_class, element_id=6):
    mask = atom_index == element_id
    equi_class = equi_class[mask]
    nmr = nmr[mask]
    unique_equi_class = np.unique(equi_class)
    return np.array([np.mean(nmr[equi_class == _class]) for _class in unique_equi_class])

def predict_nmr_from_mol(mol_list, clf=clf, nmr_type=['C','H'], raw=False, include_active_hs=True):
    if isinstance(mol_list, Chem.Mol):
        mol_list = [mol_list]
    if isinstance(mol_list[0], str):
        mol_list = [Chem.MolFromSmiles(s) for s in mol_list]
    mol_list = [Chem.AddHs(mol) for mol in mol_list]
    
    shifts_list, elements_list = predict_nmr_from_nmrnet(mol_list, clf, merge=True)
    equi_class_list = [get_equi_class(mol) for mol in mol_list]
    
    if raw:
        return {
            'mol': mol_list,
            'smiles': [Chem.MolToSmiles(Chem.RemoveHs(mol)) for mol in mol_list],
            'atoms_shift': shifts_list,
            'atoms_element': elements_list,
            'atoms_equi_class': equi_class_list,
        }

    H_nmr = []
    C_nmr = []
    for mol, shifts, elements, equi_class in zip(mol_list, shifts_list, elements_list, equi_class_list):
        if 'H' in nmr_type:
            if not include_active_hs:
                active_hs = get_active_hs(mol)
                elements[active_hs] = 0
            H_nmr.append(np.sort(shifts[elements == 1]))
        if 'C' in nmr_type:
            C_nmr.append(np.sort(merge_equi_nmr(shifts, elements, equi_class, element_id=6)))

    return {'H_nmr': H_nmr, 'C_nmr': C_nmr}


def process_merge_equi_nmr(args):
    shifts, elements, equi_class = args
    return np.sort(merge_equi_nmr(shifts, elements, equi_class, element_id=6))


def get_nmr_lists(shifts_list, elements_list, equi_classes_list, sorted=True, only_h=False, p=None):
    H_nmr_list = [shifts[elements == 1] for shifts, elements in zip(shifts_list, elements_list)]
    H_nmr_list = [H_nmr[~np.isnan(H_nmr)] for H_nmr in H_nmr_list]
    if sorted:
        H_nmr_list = [np.sort(H_nmr) for H_nmr in H_nmr_list]
    
    if only_h:
        return H_nmr_list, None
    
    if p:
        C_nmr_list = p.map(process_merge_equi_nmr, zip(shifts_list, elements_list, equi_classes_list))
    else:
        C_nmr_list = [np.sort(merge_equi_nmr(shifts, elements, equi_class, element_id=6))
                      for shifts, elements, equi_class in zip(shifts_list, elements_list, equi_classes_list)
                      ]
    
    return H_nmr_list, C_nmr_list


def get_active_hs(mol):
    """
    Get the active hydrogens in a molecule: neighboring atoms with O, N or S.
    """
    return [
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetSymbol() == 'H' and atom.GetNeighbors()[0].GetSymbol() in ['O', 'N', 'S']
    ]


if __name__ == '__main__':
    smiles_list = ['CC', 'CCO']
    mol_list = [Chem.MolFromSmiles(s) for s in smiles_list]
    mol_list = [Chem.AddHs(mol) for mol in mol_list]
    print(predict_nmr_from_mol(mol_list, raw=False))
    print(predict_nmr_from_mol(mol_list, raw=True))
    