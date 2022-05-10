# Anomaly Detection

Within this folder are all results, and utilities for the different anomaly detection projects that have been worked on. There are three different anomaly detection approaches:

## 1. Single class knowledge engineering approach

These were the first classifiers written. The classifiers themselves are in the top level folder, but inside the `Singleclass_KE_Detection` folder are the experimental results from them.

## 2. Multiclass knowledge engineering approach

After single class, multiclass classifiers were written, one for fDOM, and one for turbidity. They are in the `Multiclass_Detection` folder. The jupyter notebooks that run the multiclass detectors are in the top level directory, read `STRUCTURE.md` in the top level for a description of those files.

## 3. Deep Learning Approach

The latest approach leverages deep learning, specifically the Resnet architecture. The model, and datasets can be found in the `Deep_Learning` folder. Again, to run the classifiers look at the top level directory. There are python files and jupyter notebook files that can train and test the classifiers.
