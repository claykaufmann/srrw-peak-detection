# Repository File Structure

Initially, this project was structured using sibling folders, which Python does not play nicely with. In order to make this easier for those running it in the future, the structure has been redefined using a more Pythonic file structure. As a result, the top level of the repository is a little messier, but the actual imports within the files work much better.

## Main Files

All files that require the use of a submodule, such as th Tools submodule, are at the top level of the directory. These include all peak detection files, whether they be jupyter notebook or normal python files. In order to stop the top level from getting too cluttered, the files that do not require a submodule will stay within an inner folder.

### Deep Learning

The deep learning files in the top level directory are as follows:

#### fDOM

Two files are here:

1. `fdom_kfold_classifier_DL.py` - a python file that trains and tests the fdom deep learning peak classifier (mainly used for training on the Vermont Advanced Computing Core)
2. `fdom_kfold_classifier_DL.ipynb` - does the exact same as above, but in a Jupyter Notebook for easier use

#### Turbidity

The exact same as fDOM above, but for turbidity.

### Multiclass Knowledge Engineering Detection

There are two files, one for fDOM, and one for turbidity, both in Jupyter notebooks:

1. `multiclass_fDOM_peak_detection.ipynb`
2. `multiclass_turb_peak_detection.ipynb`

### Singleclass Knowledge Engineering Detection

These were done earlier by Zach Fogg, they were used as the baseline for designing the multiclass versions. There are 4 classifiers:

#### fDOM

1. `detect_fDOM_PLP.ipynb` - classifies peaks into either plummeting peaks, or not anomaly peaks, only given a candidates list of PLP cands
2. `detect_fDOM_PP.ipynb` - classifies peaks into either phantom peaks, or not anomaly peaks, only given a candidates list of PP cands
3. `detect_fDOM_SKP.ipynb` - classifies peaks into either skyrocketing peaks, or not anomaly peaks, only given a candidates list of SKP cands

#### Turbidity

Only one turbidity classifier of the single class exists, it is `detect_turb_PP.ipynb`, and it classifies peaks into either phantom peaks, or not anomaly peaks. Like the fDOM classifiers above, it is given a candidates list of only turbidity PP cands.
