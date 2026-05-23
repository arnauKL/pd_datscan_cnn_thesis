# Deep Learning-Based Classification of Parkinson's Disease from DaTscan Images

Source code accompanying the bachelor's thesis *Deep Learning-Based
Classification of Parkinson’s Disease from DaTscan Images*. The project
evaluates convolutional neural networks on DaTscan SPECT imaging, classical
machine learning baselines on semi-quantitative clinical variables, and
multimodal fusion approaches for Parkinson’s disease classification.

## Repository Structure

- [`src/`](src/): Shared components used across experiments:
  - `architectures.py`: custom CNN architectures
  - `transforms.py`: MONAI image transforms
  - `resnet.py`: adapted from [MedicalNet](https://github.com/Tencent/MedicalNet.git)
- [`train.py`](train.py): Main training script; loads data mappings, applies transforms, trains the configured architecture, and writes results to [`outputs/`](outputs/)
- [`classic_ml/`](classic_ml/): Scikit-learn baselines on tabular data, including a multimodal integration script
- [`analysis/`](analysis/): Post-hoc evaluation and explainability (Grad-CAM, SHAP)
- [`evaluate/`](evaluate/): Performance metrics and boxplot generation over [`outputs/`](outputs/)
- [`outputs/`](outputs/): Training results and metrics
- [`data/`](data/): Patient image path-to-label mappings; not pushed


## Dependencies

- PyTorch
- MONAI
- scikit-learn
- pandas, numpy
- SHAP, grad-cam

## Reproducibility

Experiments were developed and evaluated using stratified cross-validation
protocols. Configuration parameters, preprocessing pipelines, and evaluation
utilities are included in the repository to facilitate reproducibility.

## License

The source code is distributed under the BSD 3-Clause License. See [`LICENSE`](LICENSE) for details.

## Thesis report

The accompanying thesis manuscript, including the Typst source code and compiled
report, is available in the [companion
repository](https://github.com/arnauKL/tfg_writing)
