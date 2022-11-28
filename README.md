# Classifier to tW boosted l+jets analysis

## Installation

Setup for Maxwell Cluster at DESY

Basic instructions here: https://confluence.desy.de/display/UHHML/Using+GPU%27s+on+Maxwell

This frameworks is based upon TensorFlow 2.0 (GPU version) / Keras

After initialization of a new anaconda environment, fellow these instructions for cudatoolkit/cudnn/TF/Keras (will install python 3.9):
https://www.tensorflow.org/install/pip#linux (last accessed: 2022-11-28)
Use the pip install method for TF.

Also install these packages via `conda install -c conda-forge`:

- shap
- numpy
- matplotlib
- pandas
- uproot (don't install ROOT to avoid conflicts!)
- pickle
- scikit-learn
- To plot model architecture with Keras, you also need: pydot (via pip) and graphviz (via conda)
