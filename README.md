# Parkinson's Disease Prediction via DaTscan CNNs and Multimodal ML

Source code for my bachelor's thesis, which explores the use of CNNs on DaTscan imaging,
classical ML baselines on tabular data, and multimodal fusion for Parkinson's disease prediction.

## Repository Structure

- [`src/`](src/): Core building blocks shared across experiments:
  - `architectures.py`: custom CNN architectures
  - `transforms.py`: MONAI image transforms
  - `resnet.py`: adapted from [MedicalNet](https://github.com/Tencent/MedicalNet.git)
- [`train.py`](train.py): Main training script; loads data mappings, applies transforms, trains the configured architecture, and writes results to `outputs/`
- [`classic_ml/`](classic_ml/): Scikit-learn baselines on tabular data, including a multimodal integration script
- [`analysis/`](analysis/): Post-hoc evaluation and explainability (Grad-CAM, SHAP)
- [`evaluate/`](evaluate/): Performance metrics and boxplot generation over [`outputs/`](outputs/)
- [`outputs/`](outputs/): Training results and metrics; model weights (`.pth`) are kept local and not pushed
- [`data/`](data/): Patient image path–to–label mappings; not pushed


## Dependencies

- PyTorch
- MONAI
- scikit-learn
- pandas, numpy
- SHAP, grad-cam