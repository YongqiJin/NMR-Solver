import os
import yaml
import shutil
import pickle
from rdkit import Chem
from tqdm import tqdm
import time

from unicore.data.lmdb_dataset import LMDBDataset
from src.utils.base_logger import Logger
from src.utils.data_tools import save_dict_to_json, append_to_lmdb
from src.utils.chem_tools import get_elements_from_mol, draw_mol
from src.core.solver import run_solver


def main(config):
    exp_name = config['exp_name']
    os.makedirs(f"./result/{exp_name}", exist_ok=True)
    shutil.copy(args.config, f"./result/{exp_name}/config.yaml")
    shutil.copy(__file__, f"./result/{exp_name}/run.py")
    
    lmdb_data = LMDBDataset(config["data_path"])
    
    logger = Logger(log_name=exp_name).get_logger()
    log_path = f"./result/{exp_name}/run.log"
    if os.path.exists(log_path):
        os.remove(log_path)
    logger.handlers[-1].baseFilename = log_path
    logger.info('exp_name: %s', exp_name)
    logger.info('config: %s', config)
    logger.info('num: %d', len(lmdb_data))
    
    bit = len(str(len(lmdb_data)))
    
    all_result = {}
    iter_records = []
    time_records = []
    for id in tqdm(range(len(lmdb_data))):
        input_data = lmdb_data[id]
        H_split = input_data.get('H_split', None)
        H_shifts = input_data.get('H_shifts', None)
        C_shifts = input_data.get('C_shifts', None)
        smiles = input_data.get('smiles', None)
        
        if smiles is not None:
            mol = Chem.MolFromSmiles(smiles)
            cycle_list = mol.GetRingInfo().AtomRings()
            if cycle_list:
                config['max_cycle_length'] = max(max([len(j) for j in cycle_list]), 6)
        
        nmr_type = config['nmr_type']
        H_shifts = H_shifts if 'H' in nmr_type else None
        C_shifts = C_shifts if 'C' in nmr_type else None
        
        smiles = input_data['smiles']
        allowed_elements = get_elements_from_mol(smiles) if config['use_elements'] else None
        constraints = {"allowed_elements": allowed_elements} if config['use_elements'] else None
        candidates = input_data.get("candidates", []) if config['use_candidates'] else []
        
        logger.info('smiles: %s', smiles)
        start_time = time.time()
        result, mid_result = run_solver(config, H_split=H_split, H_shifts=H_shifts, C_shifts=C_shifts, allowed_elements=allowed_elements, 
                                        constraints=constraints, candidates=candidates, logger=logger if config['enable_logger'] else None)
        time_records.append(time.time() - start_time)
        iter_records.append(len(mid_result) - 2)
        logger.info('id: %d, iter: %d, elapsed time: %.2f seconds', id, iter_records[-1], time_records[-1])
        
        result = {
            'smiles': result.smiles_list,
            'score': result.score_list,
        }
        all_result[id] = result
        if id == 0 and os.path.exists(f"./result/{exp_name}/results.lmdb"):
            os.remove(f"./result/{exp_name}/results.lmdb")
        append_to_lmdb({str(id).zfill(bit): result}, f"./result/{exp_name}/results.lmdb")
        append_to_lmdb({str(id).zfill(bit): mid_result}, f"./result/{exp_name}/mid_results.lmdb")
        
        if config["plot_result"]:
            save_dir = f"./result/{exp_name}/figs/{id}"
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            if smiles:
                draw_mol([smiles], save_path=f"{save_dir}/target.svg", mols_per_row=5)
            for stage in mid_result:
                draw_mol(mid_result[stage]['smiles'][:config['plot_result']], save_path=f"{save_dir}/{stage}.svg", mols_per_row=5)
    
    logger.info('Average time per sample: %.2f seconds', sum(time_records) / len(time_records))
    logger.info('Average iter per sample: %.2f', sum(iter_records) / len(iter_records))
    
    save_dict_to_json(all_result, f"./result/{exp_name}/results.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./config/demo.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, "r"))
    config['exp_name'] = args.config.split('/')[-1].split('.')[0]
    main(config)
    