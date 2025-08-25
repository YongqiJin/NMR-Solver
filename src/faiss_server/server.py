from flask import Flask, request, jsonify, g
import numpy as np
import faiss
import logging
import time
from typing import Dict, List, Tuple, Optional

from src.utils.base_logger import Logger

app = Flask(__name__)

class FaissServer:
    def __init__(self):
        self.indices: Dict[str, faiss.Index] = {}
        self.index_paths: Dict[str, str] = {}  # Record index paths

    def load_index(self, index_name: str, index_path: str) -> bool:
        """
        Load a FAISS index from a file.

        Args:
            index_name (str): Name of the index.
            index_path (str): Path to the index file.

        Returns:
            bool: True if the index is successfully loaded.
        """
        logger.info(f"Loading index: {index_path}")
        self.indices[index_name] = faiss.read_index(index_path)
        logger.info(f"Index loaded successfully: {index_name}")
        self.index_paths[index_name] = index_path
        return True

    def search(self, index_name: str, vectors: List[np.ndarray], k: int = 100, ef_search: int = 200) -> Tuple[Optional[List[List[float]]], Optional[List[List[int]]]]:
        """
        Perform a search on the specified FAISS index.

        Args:
            index_name (str): Name of the index.
            vectors (List[List[float]]): Query vectors.
            k (int, optional): Number of nearest neighbors to retrieve. Defaults to 100.
            ef_search (int, optional): Search parameter for HNSW. Defaults to 200.

        Returns:
            Tuple[Optional[List[List[float]]], Optional[List[List[int]]]]: Distances and indices of the nearest neighbors.
        """
        if index_name not in self.indices:
            # If the index is not loaded, try to load it from the path
            if index_name in self.index_paths:
                self.load_index(index_name, self.index_paths[index_name])
            else:
                return None, None

        if hasattr(self.indices[index_name], 'hnsw'):
            self.indices[index_name].hnsw.efSearch = ef_search

        vectors = np.array(vectors).astype(np.float32)
        D, I = self.indices[index_name].search(vectors, k)
        return D.tolist(), I.tolist()

server = FaissServer()

@app.route('/load', methods=['POST'])
def load_index():
    """
    Flask route to load a FAISS index.

    Request JSON:
        {
            "index_name": str,
            "index_path": str
        }

    Returns:
        JSON: {"success": bool}
    """
    data = request.json
    index_name = data.get('index_name')
    index_path = data.get('index_path')
    
    success = server.load_index(index_name, index_path)
    return jsonify({"success": success})

@app.route('/search', methods=['POST'])
def search():
    """
    Flask route to perform a search on a FAISS index.

    Request JSON:
        {
            "index_name": str,
            "vectors": List[List[float]],
            "k": int (optional),
            "ef_search": int (optional)
        }

    Returns:
        JSON: {"D": List[List[float]], "I": List[List[int]]}
    """
    data = request.json
    index_name = data.get('index_name')
    vectors = np.array(data.get('vectors')).astype(np.float32)
    k = data.get('k', 1000)
    ef_search = data.get('ef_search', 2*k)
    
    D, I = server.search(index_name, vectors, k, ef_search)
    return jsonify({"D": D, "I": I})

@app.before_request
def before_request():
    """
    Record the start time of the request for logging.
    """
    g.start_time = time.time()
    
@app.after_request
def after_request(response):
    """
    Log the execution time of the request.

    Args:
        response: The Flask response object.

    Returns:
        The Flask response object.
    """
    duration = time.time() - g.start_time
    # record the endpoint name (function name) and request path
    endpoint = request.endpoint or "unknown_endpoint"
    path = request.path
    method = request.method
    logger.info(f"Request [{method}] {path} (endpoint: {endpoint}) Execution Time: {duration:.4f} seconds")
    return response

if __name__ == '__main__':
    logger = Logger(log_name='server').get_logger()
    logger.setLevel(logging.INFO)
    
    # Preload indices
    preloaded_indices = {
        "HC": "./database/index/HC.index",
        # "H": "./database/index/H.index",
        # "C": "./database/index/C.index",
    }
    for index_name, index_path in preloaded_indices.items():
        server.load_index(index_name, index_path)
    
    app.run(host='0.0.0.0', port=5000)
