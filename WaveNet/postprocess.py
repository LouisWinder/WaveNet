"""
Author: Louis Winder

Class containing post-processing operations for output audio.
"""

import torchaudio

# Perform inverse mu-law quantisation on generated audios
def inverse_mu_law_transformation(audio, quantise_channels):
    return torchaudio.functional.mu_law_decoding(audio, quantise_channels)

# Save post-processed output to file
def saveToFile(postProcessed, path):
    sample_rate = 16000
    torchaudio.save(path, postProcessed, sample_rate, encoding="PCM_S", bits_per_sample=16)
    print("Post-processed output saved to file!")
