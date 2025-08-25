# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import lmdb
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import json
import numpy as np
import os
from typing import Dict, List, Union


def write_lmdb(data: Union[Dict, List], path: str):
    """
    Writes data to an LMDB database.
    """
    try:
        os.remove(path)
        print("Remove existing lmdb: {}".format(os.path.abspath(path)))
    except:
        pass
    env_new = lmdb.open(
        path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        # max_readers=1,
        map_size=int(1e12),
    )
    txn_write = env_new.begin(write=True)
    i = 0
    if isinstance(data, list):
        num = len(data)
        bit = len(str(num))  # Ensure enough digits to represent all keys
        for index in tqdm(range(len(data))):
            inner_output = pickle.dumps(data[index], protocol=-1)
            txn_write.put(str(i).zfill(bit).encode(), inner_output)
            i += 1
            if i % 100 == 0:
                txn_write.commit()
                txn_write = env_new.begin(write=True)
        txn_write.commit()
        env_new.close()
    elif isinstance(data, dict):
        for key in tqdm(data.keys()):
            inner_output = pickle.dumps(data[key], protocol=-1)
            txn_write.put(key, inner_output)
            i += 1
            if i % 100 == 0:
                txn_write.commit()
                txn_write = env_new.begin(write=True)
        txn_write.commit()
        env_new.close()
    else:
        raise ValueError("Data type not supported: {}".format(type(data)))
    
    print("Write to lmdb: {}".format(os.path.abspath(path)))


def append_to_lmdb(data: Dict, path: str):
    """
    Appends new {key: value} pairs to an existing LMDB database.
    """
    env = lmdb.open(
        path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=int(1e12),
    )
    txn_write = env.begin(write=True)
    i = 0

    for key, value in tqdm(data.items()):
        key_bytes = key.encode() if isinstance(key, str) else key
        inner_output = pickle.dumps(value, protocol=-1)
        txn_write.put(key_bytes, inner_output)
        i += 1
        if i % 100 == 0:
            txn_write.commit()
            txn_write = env.begin(write=True)

    txn_write.commit()
    env.close()

    print("Appended to lmdb: {}".format(os.path.abspath(path)))


def query_batch_keys(path: str, keys_batch: List) -> List:
    """
    Queries a batch of keys from an LMDB database.
    """
    # Each process creates its own LMDB environment
    env = lmdb.open(
        path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    txn_read = env.begin(write=False)
    
    results = []
    for k in keys_batch:
        key_bytes = k.encode() if isinstance(k, str) else k
        value = txn_read.get(key_bytes)
        if value:
            results.append(pickle.loads(value))
        else:
            results.append(None)
    
    env.close()
    return results


def query_lmdb(path: str, keys: Union[List, str]) -> List:
    """
    Queries keys from an LMDB database.
    """
    # Handle single key case
    if not isinstance(keys, List):
        env = lmdb.open(
            path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        txn_read = env.begin(write=False)
        key_bytes = keys.encode() if isinstance(keys, str) else keys
        value = txn_read.get(key_bytes)
        env.close()
        return pickle.loads(value) if value else None
    
    # Process keys in batches
    num_processes = 8
    batch_size = (len(keys) - 1) // num_processes + 1
    keys_batches = [keys[i:i + batch_size] for i in range(0, len(keys), batch_size)]
    
    # Each process handles a batch of keys
    with Pool(num_processes) as p:
        results_batches = p.map(partial(query_batch_keys, path), keys_batches)
    
    # Merge results
    results = []
    for batch in results_batches:
        results.extend(batch)
    
    return results


def query_num_entries_lmdb(path: str) -> int:
    """
    Queries the number of entries in an LMDB database.
    """
    env = lmdb.open(
        path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin() as txn:
        stats = txn.stat()
        num_entries = stats['entries']
    return num_entries


def save_dict_to_json(data_dict: Dict, filename: str):
    """
    Saves a dictionary to a JSON file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
            
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

