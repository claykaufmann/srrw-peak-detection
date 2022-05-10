# Deep Learning

This folder contains Jupyter Notebooks, and utilities for the Deep Learning segment of the peak detection project.

## File Structure

`datasets.py`: the pytorch datasets for use with the deep learning classifiers
`resnet.py`: the 1D resnet implementation for use with deep learning

### Jupyter Notebooks

The `fdom.ipynb` and `turb.ipynb` represent very rudimentary attempts at deep learning for peak detection. They leverage a simple train/test split of 85% train data, 15% test data. The main deep learning classifiers use KFold cross validation, and as such those should be relied upon for actual classification. These two files are here simply for testing purposes. They may or may not function correctly.
