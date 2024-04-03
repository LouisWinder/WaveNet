import torchaudio
import torch
import os

# Splits long audios into subsets of smaller audios
# IMPORTANT: Files MUST be .wav format
# splitDuration : the number of SAMPLES per output (8kHz = 8000 samples / second)
# Audio will be split into n sub-audios each of length splitDuration
# where n = floor(total samples / splitDuration)

# Loads audio files into array from folder path
def loadAudios(path):
    audios = os.listdir(path)
    return audios

# Loads audio from given (wav) file
def loadAudio(folder, file):
    audio, _ = torchaudio.load("{}/{}".format(folder, file))
    return audio

def split(audios, splitDuration, folder):
    subsets = []
    for a in audios:
        audio = loadAudio(folder, a)
        # Determine number of subsets from required length of subsets
        n = audio.size(dim=1) // splitDuration
        subset = torch.split(audio, splitDuration, dim=1)
        subsets.append(subset)
    return subsets

def saveAudios(subsets):
    i = 0
    for subset in subsets:
        for n in subset:
            path = "split_output/{}.wav".format(i)
            torchaudio.save(path, n, 16000, encoding="PCM_S", bits_per_sample=16)
            i += 1


folder = "dataset"
audios = loadAudios(folder)
splitDuration = 80000
subsets = split(audios, splitDuration, folder)
saveAudios(subsets)