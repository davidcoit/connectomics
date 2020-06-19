# connectomics
This code contains two files used to build a prediction model on data from the 2014 Kaggle Connectomics challenge. Briefly, the goal of the challenge was to source methods for identifying neural connections from pre-segmented calcium fluorescence imagine data.

The file augment.py was used to pre-process fluorescence files, including running denoising and spike deconvolution algorithms using the CaImAn package and an implementation of network activity-dependent signal weighting as described in Sutera et al [2014].

The file "connectomics_rnn_cnn.ipynb" is a Jupyter notebook containing my construction of a deep learning predictive model that combines custom RNN and CNN modules built using PyTorch. The CNN are loosely based on Google's InceptionNet, but are adapted to deal with "rectangular" time series data.

All code was run on NYU's Prince HPC facility.

References:

Kaggle Connectomics Challenge: https://www.kaggle.com/c/connectomics/

Sutera, Antonio, Arnaud Joly, Vincent François-Lavet, Zixiao Aaron Qiu, Gilles Louppe, Damien Ernst, and Pierre Geurts. 2014. “Simple Connectome Inference from Partial Correlation Statistics in Calcium Imaging.” ​ArXiv:1406.7865 [Cs, Stat],​ November. http://arxiv.org/abs/1406.7865​.
