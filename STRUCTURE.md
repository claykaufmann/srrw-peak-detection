# Repository File Structure

Initially, this project was structured using sibling folders, which Python does not play nicely with. In order to make this easier for those running it in the future, the structure will be redefined using a more Pythonic file structure.

## Main Files

All files that require the use of a submodule, such as th Tools submodule, are at the top level of the directory. These include all peak detection files, whether they be jupyter notebook or normal python files. In order to stop the top level from getting too cluttered, the files that do not require a submodule will stay within an inner folder.
