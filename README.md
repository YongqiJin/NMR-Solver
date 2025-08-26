# NMR-Solver

[![GitHub](https://img.shields.io/badge/GitHub-NMR--Solver-6e7681?logo=github)](https://github.com/YongqiJin/NMR-Solver)
[![arXiv](https://img.shields.io/badge/arXiv-2508.XXXXX-b31b1b?logo=arxiv)](https://arxiv.org/abs/2508.XXXXX)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-SimNMR--PubChem-ffc107?logo=huggingface)](https://huggingface.co/datasets/yqj01/SimNMR-PubChem)
[![Zenodo](https://img.shields.io/badge/Zenodo-NMR--Solver-28a745?logo=zenodo)](https://doi.org/10.5281/zenodo.16952024)
[![Docker](https://img.shields.io/badge/Docker-nmr__solver-2496ed?logo=docker)](https://hub.docker.com/r/yqjin/nmr_solver)
[![Borhium](https://img.shields.io/badge/Borhium%20App-NMR--Toolbox-7B1FA2?logo=https://raw.githubusercontent.com/YongqiJin/NMR-Solver/main/assets/bohrium.png)](https://www.bohrium.com/apps/nmr-toolbox)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **NMR-Solver: Automated Molecular Structure Elucidation via Large-Scale Spectra Matching and Physics-Guided Fragment Optimization**

## Overview

This project integrates large-scale spectral matching with physics-guided fragment optimization, providing a powerful framework for automated molecular structure elucidation from <sup>1</sup>H and <sup>13</sup>C NMR spectra.

![framework](assets/framework.png)
![FB-MO](assets/FB-MO.png)

## Setup

### SimNMR-PubChem Database

The processed dataset (373 GB) and database index (128 GB) for the SimNMR-PubChem Database are available on [Hugging Face](https://huggingface.co/datasets/yqj01/SimNMR-PubChem). Please place them in the [database](database) directory.

### Models & Datasets

Pre-trained model weights and evaluation datasets can be downloaded on [Zenodo](https://doi.org/10.5281/zenodo.16952024). Please place them in the [model](model) and [data](data) directories respectively.

### Docker

You can use our pre-built Docker image for easy setup and deployment:

```bash
# Pull the latest Docker image
docker pull yqjin/nmr_solver:0.0.1

# Run the container interactively
docker run -it --rm yqjin/nmr_solver:0.0.1

# Or run with volume mounting for your data
docker run -it --rm -v /path/to/your/data:/workspace/data yqjin/nmr_solver:0.0.1
```

This Docker image includes all necessary dependencies and a pre-configured environment for running NMR-Solver.

## Usage

### Deploy Database

To deploy the SimNMR-PubChem Database, run the following command:

```bash
python src/faiss_server/server.py
```

Then update the server configuration in [config.yaml](src/faiss_server/config.yaml) to point to your server address.

### Run Demo

Modify the configuration file `config/demo.yaml` as needed.

To run the algorithm, use the following command:

```bash
sh scripts/run.sh demo
```

For evaluation, use:

```bash
sh scripts/eval.sh demo
```

## Contact Us

1. **GitHub Issues**  
   For bug reports, feature requests, or technical questions, please open an issue on our [GitHub repository](https://github.com/YongqiJin/NMR-Solver).

2. **WeChat**  
   Join our WeChat user community to discuss NMR-Solver with other users and developers. Scan the QR code below to join our group:

   ![WeChat Group QR Code](assets/wechat_group_qr.png)

3. **Email**  
   For collaboration inquiries, commercial licensing, or in-depth communication with our development team, please contact us at: [jinyongqi@dp.tech](mailto:jinyongqi@dp.tech)

## Citation

Please kindly cite our paper if you use this codebase:

```bibtex
@article{jin2025nmrsolver,
  title={NMR-Solver: Automated Molecular Structure Elucidation via Large-Scale Spectra Matching and Physics-Guided Fragment Optimization},
  author={Yongqi Jin and Junjie Wang and Fanjie Xu and Xiaohong Ji and Zhifeng Gao and Linfeng Zhang and Guolin Ke and Rong Zhu and Weinan E},
  journal={arXiv preprint arXiv:2508.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for additional details.
