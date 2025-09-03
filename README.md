# Official implementation of **NMR-Solver**

<!-- [![GitHub](https://img.shields.io/badge/GitHub-NMR--Solver-6e7681?logo=github)](https://github.com/YongqiJin/NMR-Solver) -->
[![arXiv](https://img.shields.io/badge/arXiv-2509.00640-b31b1b?logo=arxiv)](https://arxiv.org/abs/2509.00640)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-SimNMR--PubChem-ffc107?logo=huggingface)](https://huggingface.co/datasets/yqj01/SimNMR-PubChem)
[![Zenodo](https://img.shields.io/badge/Zenodo-NMR--Solver-28a745?logo=zenodo)](https://doi.org/10.5281/zenodo.16952024)
[![Borhium](https://img.shields.io/badge/Borhium%20App-NMR%20Toolbox-2496ed)](https://www.bohrium.com/apps/nmr-toolbox)
<!-- <a href="https://bohrium.com" style="position:relative; display:inline-block;">
  <img src="https://img.shields.io/badge/Borhium%20App-NMR%20Toolbox-2496ed?logo=zenodo"
       alt="Borhium" />
  <img src="https://raw.githubusercontent.com/YongqiJin/NMR-Solver/main/assets/bohrium.svg"
       width="14" height="14"
       style="position:absolute; left:5px; top:11%; opacity:0.8;" />
</a> -->


# NMR-Solver: Automated Structure Elucidation via Large-Scale Spectra Matching and Physics-Guided Fragment Optimization

## Overview

This project integrates large-scale spectral matching with physics-guided fragment optimization, providing a powerful framework for automated molecular structure elucidation from <sup>1</sup>H and <sup>13</sup>C NMR spectra.

<img src="assets/framework.png" style="width: 100%;" alt="framework">
<img src="assets/FB-MO.png" style="width: 100%;" alt="FB-MO">

## Online App

For the most seamless experience, try our web-based application directly without any installation:

🚀 **[Try NMR-Toolbox on Bohrium](https://www.bohrium.com/apps/nmr-toolbox)**

Hosted on the Bohrium platform, NMR-Toolbox offers an intuitive interface for:
- **NMR Database Search**
- **Structure Elucidation from NMR**
- **Chemical Shift Prediction & Spectral Matching**


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

<div align="center">
<img src="assets/wechat_group_qr.png" style="width: 50%;" alt="WeChat Group QR Code">
</div>

3. **E-mail**  
   For collaboration inquiries, commercial licensing, or in-depth communication with our development team, please contact us at: [jinyongqi@dp.tech](mailto:jinyongqi@dp.tech)

## Citation

Please kindly cite our paper if you use this codebase:

```bibtex
@article{jin2025nmrsolver,
  title={NMR-Solver: Automated Structure Elucidation via Large-Scale Spectral Matching and Physics-Guided Fragment Optimization},
  author={Jin, Yongqi and Wang, Jun-Jie and Xu, Fanjie and Ji, Xiaohong and Gao, Zhifeng and Zhang, Linfeng and Ke, Guolin and Zhu, Rong and E, Weinan},
  year={2025},
  journal={arXiv preprint arXiv:2509.00640}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for additional details.
