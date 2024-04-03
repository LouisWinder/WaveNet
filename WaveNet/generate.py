"""
Author: Louis Winder

Class used to generate audio samples using a trained WaveNet model.
"""

import torch
import numpy as np
from Wavenet import WaveNet
from preprocess import *
from postprocess import *
import random
import math
import torchaudio

# Load a pre-trained model to be used for generation
def load_model(path):
    wavenet = WaveNet()
    gpu = "cuda:0"
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    trained = torch.load(path, map_location=device)
    wavenet.load_state_dict(trained["model_state_dict"])
    wavenet.eval() # Ensure WaveNet model is in "evaluation/inference" mode
    return wavenet

# Generate using the pre-trained model
def generate(scheme, num_samples):
    model = load_model("model.pth") # Load trained model
    gpu = "cuda:0"
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Choose generation scheme
    if scheme == "Seeding":
	# If using seeding scheme, provide initial seed length (in samples)
        seed_samples = initialiseSeed(16000)
        return generate_seeding(model, num_samples, seed_samples, device)
    elif scheme == "Unique":
        return generate_unique(model, num_samples, device)

# Choose sample from output probability distribution that will be the chosen sample
def choose_sample(scheme, sample_distribution):
    if scheme == "Highest":
        # Choose sample with highest probability
        l = sample_distribution.size(dim=1)
        l -= 1
        sample = torch.argmax(sample_distribution, dim=2)[0, l]
        return sample
    else:
        # Choose random sample from probability distribution (within reasonable range)
        # Find the top probabilities
        l = sample_distribution.size(dim=1)
        l = l - 1
        nextSampleProb = sample_distribution[0, l]
        nextSampleProb = nextSampleProb.detach().cpu().numpy()
        # Randomly sample from softmax probability distribution (this produces best results)
        return np.random.choice(np.arange(256), p=nextSampleProb)

# Initialise the seed waveform for input into the model via "generate_seeding"
def initialiseSeed(seed_samples):
    # For best results, Initialise seed using waveform similar (not the same) as one the model has been trained on
    # Get seed example
    file = "example.wav"
    seed, _ = torchaudio.load(file)
    # Take "seed_samples" samples from seed example
    seed = torch.split(seed, seed_samples, dim=1)[0]
    # Mu-law encode the seeded waveform
    return mu_law_transformation(seed)

# Generates samples using the "seeding" method
# Given an initial waveform seed (list of samples),
# let the model predict the next (num_samples - seed_samples)
# samples.
def generate_seeding(model, num_samples, seed_samples, device):
    # Number of seed samples
    num_seed_samples = seed_samples.size(dim=1)
    # Number of samples left to generate
    to_generate = num_samples - num_seed_samples
    # Pre-allocate space for predicted samples
    samples = torch.zeros((1, num_samples), dtype=torch.int16)
    # Provide initial seed
    samples[0][:num_seed_samples] = seed_samples
    for i in range(to_generate - 1):
        currentlyAnalysing = torch.slice_copy(samples, dim=1, start=0, end=(num_seed_samples+i))
        # One-hot encode samples for input into model
        sample = one_hot_encoding(currentlyAnalysing)
        sample = torch.permute(sample, (0, 2, 1))
        sample = sample.type(torch.FloatTensor)
        sample = sample.to(device)
	# Predict the next samples
        sample = model(sample)
	# Compute the softmax of the output
        sample = torch.softmax(sample, dim=2)
        # Choose next sample from predicted probability distribution
        sample = choose_sample("Random", sample)
	# Append chosen next sample to previous chosen samples
        samples[0, (num_seed_samples+i)] = sample
        #print("Samples generated: {}/{} ({}% generated)".format(i + 1, to_generate - 1, (i / to_generate) * 100))
    # Collate predicted samples into single waveform; perform inverse mu-law transformation
    inverseCompanded = inverse_mu_law_transformation(samples, 256)
    inverseCompanded = inverseCompanded.detach()
    return inverseCompanded

# Generates samples using the "unique" method
# Given an initial random sample in range 0 to (quantisation_channels - 1), let
# the model produce entirely unique data.
def generate_unique(model, num_samples, device):
    # Generate random initial sample
    randomSample = random.randint(0, 255)
    #randomSample = 128
    samples = torch.zeros((1, num_samples), dtype=torch.int16) # Collection of predicted samples
    samples[0, 0] = randomSample
    for i in range(num_samples-1):
        currentlyAnalysing = torch.slice_copy(samples, dim=1, start=0, end=i+1)
        # One-hot encode sample for input into model
        randomSample = one_hot_encoding(currentlyAnalysing)
        randomSample = torch.permute(randomSample, (0, 2, 1))
        randomSample = randomSample.type(torch.FloatTensor)
        randomSample = randomSample.to(device)
	# Predict the next samples
        randomSample = model(randomSample)
	# Compute the softmax of the output
        randomSample = torch.softmax(randomSample, dim=2)
        # Choose sample from predicted probability distribution
        randomSample = choose_sample("Random", randomSample)
	# Append chosen next sample to previous chosen samples
        samples[0, i+1] = randomSample
        #print("Samples generated: {}/{} ({}% generated)".format(i+1, num_samples-1, i / num_samples * 100))
    # Collate predicted samples into single waveform; perform inverse mu-law transformation
    inverseCompanded = inverse_mu_law_transformation(samples, 256)
    inverseCompanded = inverseCompanded.detach()
    return inverseCompanded

# num_samples = number of samples to generate
# Seconds = num_samples / sampling frequency
generated = generate("Unique", 80000)
# Save generated waveform to file
saveToFile(generated, "generated.wav")
