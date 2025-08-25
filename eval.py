import os
import argparse
from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool
from functools import partial
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict
from scipy.stats import wasserstein_distance as wd

from unicore.data.lmdb_dataset import LMDBDataset

from src.utils.chem_tools import get_elements_from_mol, tanimoto_morgan_similarity
from src.utils.data_tools import save_dict_to_json
from src.core.constraint import satisfy_constraints


def mol_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles)

def get_canonical_smiles(smiles, strict=True):
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smiles)), isomericSmiles=strict)

def eval_topk_recall(prediction, target, topk_list):
    return [1 if target in prediction[:topk] else 0 for topk in topk_list]

def eval_topk_tanimoto(p, prediction, target, topk_list, useChirality):
    if len(prediction) == 0:
        return [0] * len(topk_list)
    tanimoto_similarity = p.map(partial(tanimoto_morgan_similarity, mol2=target, useChirality=useChirality), prediction[:max(topk_list)])
    return [max(tanimoto_similarity[:topk]) for topk in topk_list]


def wasserstein_distance(x: np.ndarray, y: np.ndarray):
    if len(x) == 0 or len(y) == 0:
        return float('inf')
    return wd(x, y)


def evaluate_iter(targets: List[str], predictions: Dict[int, List[str]], output_file):
    
    topk_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000]
    columns = ['ID'] + ['rank'] + [f'recall@{topk}' for topk in topk_list] + [f'tanimoto@{topk}' for topk in topk_list] + \
        ['rank(soft)'] + [f'recall@{topk}(soft)' for topk in topk_list] + [f'tanimoto@{topk}(soft)' for topk in topk_list]
    eval_results = pd.DataFrame(columns=columns)
    
    with Pool(16) as p:
        for id in tqdm(range(len(targets))):
            origin_target = targets[id]
            try:
                origin_prediction = predictions[id][:max(topk_list)]
            except:
                print(f"Warning: ID {id} not found in mid_results")
                continue
            
            prediction = p.map(partial(get_canonical_smiles, strict=True), origin_prediction)
            target = get_canonical_smiles(origin_target, strict=True)
            rank_strict = prediction.index(target) + 1 if target in prediction else -1
            recall_strict = eval_topk_recall(prediction, target, topk_list)
            tanimoto_strict = eval_topk_tanimoto(p, prediction, target, topk_list, useChirality=True)
            
            prediction = p.map(partial(get_canonical_smiles, strict=False), origin_prediction)
            target = get_canonical_smiles(origin_target, strict=False)
            rank_soft = prediction.index(target) + 1 if target in prediction else -1
            recall_soft = eval_topk_recall(prediction, target, topk_list)
            tanimoto_soft = eval_topk_tanimoto(p, prediction, target, topk_list, useChirality=False)
        
            eval_results.loc[len(eval_results)] = [id] + [rank_strict] + recall_strict + tanimoto_strict + [rank_soft] + recall_soft + tanimoto_soft
    
    eval_results['ID'] = eval_results['ID'].astype('int')
    eval_results = eval_results.astype({col: 'int' for col in columns if col.startswith('recall')})
    eval_results.to_csv(output_file, index=False)
    
    avg_results = eval_results.mean().to_dict()
    del avg_results['ID']
    avg_results['rank'] = eval_results.loc[eval_results['rank'] > -1, 'rank'].mean()
    avg_results['rank(soft)'] = eval_results.loc[eval_results['rank(soft)'] > -1, 'rank(soft)'].mean()
    avg_results['num_eval'] = len(eval_results)
    avg_results['num_total'] = len(targets)
    save_dict_to_json(avg_results, output_file.replace('.csv', '.json'))
    

def evaluate(result_dir, constraints_list):
    
    config = yaml.safe_load(open(os.path.join(result_dir, 'config.yaml')))
    pred_file = os.path.join(result_dir, 'mid_results.lmdb')
    target_file = config['data_path']
    
    targets = LMDBDataset(target_file)
    mid_results = LMDBDataset(pred_file)
    
    print(f"targets: {len(targets)}, mid_results: {len(mid_results)}")
    
    output_eval_dir = os.path.join(os.path.dirname(pred_file), "eval")
    os.makedirs(output_eval_dir, exist_ok=True)
    constraints_str =  '_'.join(constraints_list) if constraints_list else 'no'
    
    constraints_map = {
        "elements": lambda x: get_elements_from_mol(x), 
        "formula": lambda x: Chem.rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(x))
    }
    # Filter molecules in mid_results that do not satisfy the constraints
    mid_results_copy = {i: mid_results[i] for i in range(len(mid_results))}
    if constraints_list:
        for k in constraints_list:
            if k not in constraints_map:
                raise ValueError(f"Invalid constraint: {k}. Available constraints: {list(constraints_map.keys())}")
    else:
        constraints_list = []
    print('constraints:', constraints_list)
    
    with Pool(16) as p:
        for i in tqdm(range(len(mid_results_copy))):
            item = mid_results_copy[i]
            constraints = {k: constraints_map[k](targets[i]['smiles']) for k in constraints_list}
            for stage in item:
                mol_list = p.map(mol_from_smiles, item[stage]['smiles'])
                is_satisfied = p.map(partial(satisfy_constraints, constraints=constraints), mol_list)
                filter_ids = [i for i, satisfied in enumerate(is_satisfied) if satisfied]
                for key in item[stage]:
                    item[stage][key] = [item[stage][key][i] for i in filter_ids]
    save_dict_to_json(mid_results_copy, os.path.join(result_dir, f"mid_results_filtered/{constraints_str}.json"))
    
    max_iter = config['max_iter']
    for n_iter in range(max_iter + 1):
        output_eval_file = os.path.join(output_eval_dir, f"iter_{n_iter}_{constraints_str}.csv")
        predictions_smi = {key: value[min(max(value.keys()), n_iter)]['smiles'] for key, value in mid_results_copy.items()}
        targets_smi = [item['smiles'] for item in targets]
        evaluate_iter(targets_smi, predictions_smi, output_eval_file)
        

def main():
    parser = argparse.ArgumentParser(description='Evaluate prediction results')
    parser.add_argument('--result-dir', '-d', type=str, default='./result/demo/', help='Directory containing prediction results')
    parser.add_argument('--constraints-list', '-c', nargs='+', help='List of constraints')
    
    args = parser.parse_args()
    evaluate(args.result_dir, args.constraints_list)


if __name__ == "__main__":
    main()
