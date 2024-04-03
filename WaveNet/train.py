"""
Author: Louis Winder

Class for training of a WaveNet model. Training can be done from scratch or continued using
pre-saved models.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Wavenet import WaveNet
from preprocess import one_hot_encoding
import torch
import math

# Load a pre-trained model to be used for generation
def loadPreTrained(model, optimiser, scheduler, path):
    cp = torch.load(path)
    model.load_state_dict(cp["model_state_dict"])
    optimiser.load_state_dict(cp["optimiser_state_dict"])
    scheduler.load_state_dict(cp["scheduler_state_dict"])
    return True

# Get the current learning rate (for saving current model checkpoint)
def getLearningRate(optimiser):
    for group in optimiser.param_groups:
        return group["lr"]

# Save the current model state (for resuming later from model checkpoint)
def saveModel(model, optimiser, scheduler, epoch, loss, learning_rate, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "learning_rate": learning_rate
    }, path)

# Get the last epoch from pre-trained model checkpoint (for resuming model training)
def getLastEpoch(path):
    cp = torch.load(path)
    return cp["epoch"] + 1 # +1 to start from next epoch

# Get the last learning rate from pre-trained model checkpoint (for resuming model training)
def getLastLearningRate(path):
    cp = torch.load(path)
    return cp["learning_rate"]

# Class defining the entire trining/valiation cycle for the WaveNet model
class Trainer():
    # Train the model
    def train(self, trainDataLoader, validationDataLoader, learning_rate, epochs):
        # Training stuff here...
        # Remember to save the model parameters when required training done

        gpu = "cuda:0"
        device = torch.device(gpu if torch.cuda.is_available() else "cpu")

        wavenet = WaveNet()
        wavenet = wavenet.to(device)

        lossFunction = torch.nn.CrossEntropyLoss()
        optimiser = torch.optim.AdamW(wavenet.parameters(), learning_rate)
        # Decrease learning rate by gamma every step_size epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=5, gamma=1)

        # Load pre-trained model and optimizer parameters (OPTIONAL)
        #isPreTraining = loadPreTrained(wavenet, optimiser, scheduler, "pretrained_model.pth")
        isPreTraining = False

        epochAverages = [] # Training loss averages over the epochs
        valAverages = [] # Validation loss averages over the epochs
        learning_rates = [] # Collection of learning rates over time (used if variable learning rate)
        lastEpoch = 0
        epochs_list = [] # List of all epochs ran (for plotting)
        # Number of training data
        trainDataLength = len(trainDataLoader)
        # Number of validation data
        valDataLength = len(validationDataLoader)
        if isPreTraining == True:
            # Get training epoch to resume from
            lastEpoch = getLastEpoch("pretrained_model.pth")
        for epoch in range(epochs):
            iteratorEpoch = epoch
            lossAvg = 0 # Average loss for epoch
            lossAvgVal = 0 # Average validation loss for epoch
            epoch += lastEpoch
            #print("----------EPOCH {}----------".format(epoch+1))
            epochs_list.append(epoch+1)
            for step, (trainData, trainLabel) in enumerate(trainDataLoader):
                #if step % trainDataLegth == 0:
                    #print("-----TRAINING-----")
                wavenet.train() # Ensure model is in "training" mode
                # One-hot encode audio for input into model
                trainData = one_hot_encoding(trainData)
                trainData = torch.permute(trainData, (0, 2, 1))
                trainData = trainData.type(torch.FloatTensor)
                trainData = trainData.to(device)
		# Predict the next samples using the WaveNet model
                wavenetOutput = wavenet(trainData)
                trainLabel = trainLabel.type(torch.LongTensor)
                trainLabel = trainLabel.to(device)
                wavenetOutput = torch.permute(wavenetOutput, (0, 2, 1))
		# Calculate loss between predicted output and ground truth
                loss = lossFunction(wavenetOutput, trainLabel)
                lossNum = loss.item()
		# Collate loss averages
                lossAvg += lossNum
                #print("Step: {} \nTraining Loss: {} \nLearning rate: {}".format(step+1, lossNum, getLearningRate(optimiser)))
		# Compute the gradients with respect to loss
                loss.backward()
		# Update network's weights with respect to the gradient (perform gradient descent)
                optimiser.step()
		# Zero-out gradients so they don't erroneously accumulate
                optimiser.zero_grad()
                # Validation
                # Validate training using validation data every (number of training data) steps
                with torch.no_grad():
                    if (step + 1) % trainDataLength == 0:
                        #print("\n-----VALIDATION-----")
                        wavenet.eval()
                        for validationStep, (valData, valLabel) in enumerate(validationDataLoader):
                            # One-hot encode validation data
                            valData = one_hot_encoding(valData)
                            valData = torch.permute(valData, (0, 2, 1))
                            valData = valData.type(torch.FloatTensor)
                            valData = valData.to(device)
                            wavenetOutput = wavenet(valData)
                            valLabel = valLabel.type(torch.LongTensor)
                            valLabel = valLabel.to(device)
                            wavenetOutput = torch.permute(wavenetOutput, (0, 2, 1))
                            validationLoss = lossFunction(wavenetOutput, valLabel)
                            validationLossNum = validationLoss.item()
                            lossAvgVal += validationLossNum
                            #print("Step: {} \nValidation loss: {}".format(validationStep+1, validationLossNum))
                trainAvg = math.inf
                if epoch + 1 == lastEpoch + epochs and step + 1 == trainDataLength:
                    trainAvg = lossAvg
                    trainAvg /= trainDataLength
                if trainAvg < 0.1 or (lastEpoch + epochs == epoch + 1 and step + 1 == trainDataLength):
                    if trainAvg < 0.1:
                        print("Final training loss: {}".format(trainAvg))
                        print(lossNum)
                    # Append last (current) epoch averages etc. & step scheduler
                    endOfEpochTasks(scheduler, lossAvg, trainDataLength, lossAvgVal, valDataLength,
                                    epochAverages, valAverages, iteratorEpoch, learning_rates, optimiser)
                    # Save pre-trained model and optimiser state
                    saveModel(wavenet, optimiser, scheduler, epoch, loss, getLearningRate(optimiser), "saved_model.pth")
                    #for avg in epochAverages:
                        #print("{}".format(avg))
                    initialTrainingLoss = epochAverages[0]
                    finalTrainingLoss = epochAverages[len(epochAverages)-1]
                    print("Initial TRAINING loss: {}".format(initialTrainingLoss))
                    print("Final TRAINING loss: {}".format(finalTrainingLoss))

                    initialValidationLoss = valAverages[0]
                    finalValidationLoss = valAverages[len(valAverages)-1]
                    print("Initial VALIDATION loss: {}".format(initialValidationLoss))
                    print("Final VALIDATION loss: {}".format(finalValidationLoss))

                    print("Overall TRAINING difference: {}".format(finalTrainingLoss - initialTrainingLoss))
                    print("Overall VALIDATION difference: {}".format(finalValidationLoss - initialValidationLoss))

                    # Plot training and validation losses over time
                    # Plot training loss
                    plt.plot(epochs_list, epochAverages, "r-", label="train_loss")
                    # Plot validation loss
                    plt.plot(epochs_list, valAverages, "b-", label="val_loss")
                    plt.xlabel("Epochs")
                    plt.ylabel("Loss")
                    plt.legend()
                    plt.grid()
                    # Save plot to file
                    plt.savefig("loss_plot.png") # can be image file or PDF
                    #plt.show()
                    plt.close()

                    # Plot learning rate
                    plt.plot(epochs_list, learning_rates, "r-", label="learning_rate")
                    plt.xlabel("Epochs")
                    plt.ylabel("Learning rate")
                    plt.legend()
                    plt.grid()
                    # Save plot to file
                    plt.savefig("learning_rate_plot.png")
                    #plt.show()
                    plt.close()
            # Perform end-of-epoch tasks
            endOfEpochTasks(scheduler, lossAvg, trainDataLength, lossAvgVal, valDataLength,
                            epochAverages, valAverages, iteratorEpoch, learning_rates, optimiser)
            if epoch == (epochs - 1):
                saveModel(wavenet, optimiser, scheduler, epoch, loss, getLearningRate(optimiser), "saved_model.pth")

def endOfEpochTasks(scheduler, lossAvg, trainDataLength, lossAvgVal, valDataLength, epochAverages,
                    valAverages, iteratorEpoch, learning_rates, optimiser):
    # Reduce learning rate based on validation loss (only if necessary)
    #scheduler.step(validationLossNum)
    scheduler.step()
    lossAvg /= trainDataLength
    lossAvgVal /= valDataLength
    epochAverages.append(lossAvg)
    valAverages.append(lossAvgVal)
    learning_rates.append(getLearningRate(optimiser))
    difference = 0
    if iteratorEpoch != 0:
        difference = epochAverages[iteratorEpoch] - epochAverages[iteratorEpoch - 1]
    #print("\nAverage TRAINING loss for epoch: {}\nDifference = {}".format(lossAvg, difference))
