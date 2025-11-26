# Dual-Source Metric-Based Cross-Client Membership Inference Attack in Federated Learning

Official implementation of the paper:  
**"Dual-Source Metric-Based Cross-Client Membership Inference Attack in Federated Learning"**

[[Paper]](https://arxiv.org/abs/xxxx.xxxxx) | [[Project Page]](https://github.com/yourname/DSM-MIA)

---

##  Overview

This repository provides the official implementation of **DSM-MIA**, a novel membership inference attack designed for **federated learning (FL)**.  
Unlike traditional MIAs that only determine whether a sample was used in training, DSM-MIA **identifies which specific client** in FL owns the sample.

<p align="center">
  <img src="assets/framework.png" width="700"/>
</p>

---

##  Environment Setup
`pytorch:2.0.1+cu118`
`python:3.8.17`
Before executing the project code, please prepare the Python environment according to the `requirements.txt` file. We set up the environment with python 3.8 and pytorch 2.0

##  Project Structure
```bash
DSM-MIA/
│
├── src/
│   ├── train_attack_model.py      # RNN/Transformer-based attack model
│   ├── feature_extraction.py      # Construct dual-source metric sequences
│   ├── utils.py                   # Helper functions
│
├── draw_pict/                     # Visualization scripts
├── saved_mia_models/              # Trained attack models
├── evaluate/                      # Evaluation results
├── requirements.txt
└── README.md
```
##  Quick Start

### 1.Train a target federated model
```bash
python train_fed_model.py --dataset cifar10 --model resnet18 --n_clients 5
```
### 2.Extract dual-source metrics
```bash
python feature_extraction.py --dataset cifar10 --rounds 60
```
### 3.Train the attack model
```bash
python train_attack_model.py --model RNN_Attention --epochs 50
```
### 1.Evaluate attacks
```bash
python attack_comparison.py --metric AUC --save_fig True
```
##  Results
| Dataset | Model | Attack |  **AUC** |  **TPR@FPR=0.001** |
|-------------|-----------|-----------|-------------|----------------------|
| CIFAR-10    | ResNet-18 | DSM-MIA   | **0.874**   | **0.523** |
| CIFAR-100   | ResNet-18 | DSM-MIA   | **0.812**   | **0.465** |
##  Citation

If you find our work helpful, please cite:
```bash
@article{yourname2025dsmmia,
  title={Dual-Source Metric-Based Multi-Client Membership Inference Attack in Federated Learning},
  author={Your Name and ...},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```
##  Contact

If you have questions or issues, please open an issue or contact:
## your.email@domain.com

##  License

This project is licensed under the MIT License.
