"""
Author: Louis Winder

Main driver class for model TRAINING. Calls the pre-processing function and trains the model
for specified number of epochs.
"""

from preprocess import preprocess
from train import Trainer

def main():
    # Preprocess the input
    trainDataLoader, validationDataLoader = preprocess()
    # Train the model
    trainer = Trainer()
    # NOTE: at least 2 epochs must be used (preferably at least 3 for good results)
    epochs = 25
    learning_rate = 0.001
    # Train the network
    output = trainer.train(trainDataLoader, validationDataLoader, learning_rate, epochs)

main()
