import os
import shutil
import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import pandas as pd
import time

import scipy
import scipy.signal as sgn
import scipy.misc
from sklearn.model_selection import train_test_split
from caiman.source_extraction.cnmf import deconvolution


import torch
import torch.nn as nn
import torch.functional as F


##########################################################
#####               FUNCTION DEFINITION              #####
##########################################################

def connectionMatrix(fmat, nlist):
    n = np.shape(fmat)[1]
    cmat = np.zeros((n, n))
    for connection in nlist:
        nfrom = connection[0] - 1
        nto = connection[1] - 1
        if connection[2] == 1:
            cmat[nfrom, nto] = 1
    return(cmat)


def process_flr(fmat):
    denoised = np.zeros(np.shape(fmat))
    spikes = np.zeros(np.shape(fmat))
    binspikes = np.zeros(np.shape(fmat))
    for i in range(np.shape(fmat)[1]):
        c, bl, clc, g, sn, sp, lam = deconvolution.constrained_foopsi(
            flr[:, i], p=1, method="oasis")
        denoised[:, i] = c
        spikes[:, i] = sp

    binspikes = abs(np.ceil(spikes).astype(int))
    return(denoised, spikes, binspikes)


def weight_matrix(mat, k=1):
    wmat = np.zeros(np.shape(mat))
    for i in range(np.shape(mat)[0]):
        for j in range(np.shape(mat)[1]):
            wmat[i, j] = (mat[i, j] + 1) ** (1 + (1/(0.01+np.sum(mat[i, :]))))
    return(wmat)


def pc(mat):
    n_neurons = np.shape(mat)[1]
    p = np.zeros([n_neurons, n_neurons])
    prec = scipy.linalg.inv(np.cov(mat, rowvar=False))
    for i in range(n_neurons):
        for j in range(n_neurons):
            p[i, j] = -1 * prec[i, j] / (np.sqrt(prec[i, i] * prec[j, j]))
    return(p)


def running_func(mat, func, window=3000, verbose=False):
    n_timesteps = np.shape(mat)[0]
    n_neurons = np.shape(mat)[1]
    rmat = np.zeros([n_timesteps, n_neurons, n_neurons])
    window_start = 0
    window_end = window
    for t in range(window, n_timesteps):
        window_mat = mat[window_start:window_end, :]
        window_mat_calc = func(window_mat)
        rmat[t, :, :] = window_mat_calc
        # if t % 1000 == 0 or verbose:
        #print("iterations: {}".format(window_end-window+1))
        window_start += 1
        window_end += 1
    return(rmat[:, :, :])


##########################################################
#####        SET UP FILE PATHS AND DATA IMPORT       #####
##########################################################
# parse command line arg for dataset number, get name from dataset dict
idx = sys.argv[1]

# define dataset number to dataset name dict
dataset_dict = {
    1: 'normal-1',
    2: 'normal-2',
    3: 'normal-3',
    4: 'normal-4',
    5: 'normal-3-highrate',
    6: 'normal-4-lownoise',
    7: 'lowcon',
    8: 'highcon',
    9: 'lowcc',
    10: 'highcc',
    11: 'small-1',
    12: 'small-2',
    13: 'small-3',
    14: 'small-4',
    15: 'small-5',
    16: 'small-6'
}

# define paths to data
data_parent_dir = "/scratch/dmc421/dlproject/data"
#data_parent_dir = "/Users/David/Documents/NYU/DEEP_LEARNING/finalproject/data/source/connectomics"

# dataset name : path
data_dir = {}
for v in dataset_dict.values():
    data_dir.update({v: str(os.path.join(data_parent_dir, v))})

# dataset name : path
flr_dir = {}
net_dir = {}
pos_dir = {}

for dataset in data_dir.keys():
    for f in os.listdir(data_dir[dataset]):
        filepath = os.path.join(data_dir[dataset], f)
        if "networkPositions" in f:
            pos_dir.update({dataset: filepath})
        elif "network" in f:
            net_dir.update({dataset: filepath})
        elif "fluorescence" in f:
            flr_dir.update({dataset: filepath})

# import data
x = dataset_dict[int(idx)]

fpath = flr_dir[x]
npath = net_dir[x]
ppath = pos_dir[x]

network = np.genfromtxt(npath,
                        delimiter=',').astype(int)

flr = np.genfromtxt(fpath,
                    delimiter=',')


# set filepaths for processed files
denoised_path = fpath[:-4] + "_denoised.txt"
spike_path = fpath[:-4] + "_spike.txt"
binspike_path = fpath[:-4] + "_binspike.txt"

weighted_denoised_path = fpath[:-4] + "_denoised_weighted.txt"
weighted_spike_path = fpath[:-4] + "_spike_weighted.txt"

weighted_spike_pc_path = fpath[:-4] + "_spike_weighted_pc.pt"
weighted_denoised_pc_path = fpath[:-4] + "_denoised_weighted_pc.pt"

print(denoised_path)
print(spike_path)
print(binspike_path)
print(weighted_denoised_path)
print(weighted_spike_path)
# print(weighted_spike_pc_path)
# print(weighted_denoised_pc_path)


#####################################
#####        PROCESS DATA       #####
#####################################

# process data including creating denoised, binary and non-binary spikes
n_samples = np.shape(flr)[0]
n_neurons = np.shape(flr)[1]
con = connectionMatrix(flr, network)
denoised, spikes, binspikes = process_flr(flr)

# weight denoise and spikes
weighted_denoised = weight_matrix(denoised)
weighted_spikes = weight_matrix(spikes)

# convert to float16
# denoised = denoised.astype(np.float8)
# spikes = denoised.astype(np.float8)
# binspikes = binspikes.astype(int)
# weighted_denoised = weighted_denoised.astype(np.float8)
# weighted_spikes = weighted_spikes.astype(np.float8)


# # compute partial correlation matrices for weighted flrs and spikes
# W = 1500  # memory for samples, @50Hz = 30s
#
# weighted_spike_pc = running_func(weighted_spikes, func=pc, window=W, verbose=False)
# weighted_denoised_pc = running_func(weighted_denoised, func=pc, window=W, verbose=False)
#

np.savetxt(denoised_path, denoised, fmt='%1.4f', delimiter=",")
np.savetxt(spike_path, spikes, fmt='%1.4f', delimiter=",")
np.savetxt(binspike_path, binspikes, fmt='%1.1f', delimiter=",")

np.savetxt(weighted_denoised_path, weighted_denoised, fmt='%1.4f', delimiter=",")
np.savetxt(weighted_spike_path, weighted_spikes, fmt='%1.4f', delimiter=",")

#torch.save(torch.Tensor(weighted_denoised_pc).to(torch.float16), weighted_denoised_pc_path)
#torch.save(torch.Tensor(weighted_spike_pc).to(torch.float16), weighted_spike_pc_path)
