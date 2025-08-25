import requests
import numpy as np
import yaml
import os
from typing import Tuple, Optional


class FaissClient:
    def __init__(self):
        """
        Initialize the FaissClient by loading the server URL from the configuration file.
        """
        # read the server URL from the configuration file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.yaml")
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        self.server_url = config["server_url"]
        
    def search(
        self, 
        index_name: str, 
        vectors: np.ndarray,
        k: int = 1000,
        ef_search: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a search on the FAISS server.
        """
        if ef_search is None:
            ef_search = max(2 * k, 2000)
        response = requests.post(
            f"{self.server_url}/search",
            json={
                "index_name": index_name,
                "vectors": vectors.tolist(),
                "k": k,
                "ef_search": ef_search
            }
        )
        result = response.json()
        return np.array(result["D"]), np.array(result["I"])
