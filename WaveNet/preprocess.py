"""
Author: Louis Winder

Class containing pre-processing operations for input audio and initialisation of datasets.
"""

import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Perform mu-law quantisation
def mu_law_transformation(audio):
    # Apply mu-law companding transformation, quantise to 256 values
    # To use mu-law signal must be normalised to between [-1, 1]
    mu_transformation = torchaudio.functional.mu_law_encoding(audio, 256)
    return mu_transformation.type(torch.int16)

# Perform one-hot encoding on the data
def one_hot_encoding(audio):
    oneHotEncoded = torch.nn.functional.one_hot(audio.to(torch.int64), 256)
    batchSize = audio.size(dim=0)
    oneHotEncoded = torch.reshape(oneHotEncoded, [batchSize, -1, 256])
    return oneHotEncoded

# Load the dataset from path
def loadDataset(path):
    files = os.listdir(path)
    return files

class WavenetDataset(Dataset):
    def __init__(self, training_data, labels):
        super().__init__()
        self.training_data = training_data
        self.labels = labels

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, index):
        audio = self.training_data[index]
        labels = self.labels[index]
        return audio, labels

# Pre-process the input data
def preprocess():
    # Load dataset
    path = "dataset_path"
    dataset = loadDataset(path)
    datasetLength = len(dataset)

    # Perform mu-law encoding on all data in dataset
    compandedAudios = []
    for data in dataset:
        audio, sr = torchaudio.load("dataset_path/{}".format(data))
        # Discard any audios that are not desired samples in length
        clipLength = 5 # Length of samples in seconds
        desired_samples = sr * clipLength
        if (audio.size(dim=1) != desired_samples):
            continue
        mu = mu_law_transformation(audio)
        compandedAudios.append(mu)
        #print(len(compandedAudios))
        if len(compandedAudios) > 9999: # use if only want subset of dataset
            break
    print("Number of training samples:")
    print(len(compandedAudios))

    # Split into training data and labels manually
    trainingData = []
    labels = []
    for a in compandedAudios:
        dataSize = a.size(dim=1)-1
        trainingData.append(torch.slice_copy(a, dim=1, start=0, end=dataSize))
        labels.append(torch.slice_copy(a, dim=1, start=1, end=dataSize+1))

    trainingData = torch.cat(trainingData, dim=0)
    trainingData = trainingData.type(torch.FloatTensor)
    labels = torch.cat(labels, dim=0)
    labels = labels.type(torch.FloatTensor)

    audioDataset = WavenetDataset(trainingData, labels)
    # Create generator to seed random_split (for getting same train/val split)
    generator = torch.Generator()
    generator.manual_seed(0)
    trainDataset, validationDataset = torch.utils.data.random_split(audioDataset, [0.8, 0.2], generator) ## generator
    trainDataLoader = DataLoader(trainDataset, batch_size=8, shuffle=True)
    validationDataLoader = DataLoader(validationDataset, batch_size=8, shuffle=True)
    print("Batch size: {}".format(trainDataLoader.batch_size))

    return trainDataLoader, validationDataLoader
