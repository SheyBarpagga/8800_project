import sys
# Before any more imports, leave cwd out of sys.path for internal 'conda shell.*' commands.
# see https://github.com/conda/conda/issues/6549
if len(sys.argv) > 1 and sys.argv[1].startswith('shell.') and sys.path and sys.path[0] == '':
    # The standard first entry in sys.path is an empty string,
    # and os.path.abspath('') expands to os.getcwd().
    del sys.path[0]



import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model import MultiInputModel
from dataset import phishingDataset 
import time


import torchtext; torchtext.disable_torchtext_deprecation_warning()

csv_file = "./data.csv"
batch_size = 32
num_epochs = 10
learning_rate = 0.001
# print("test")
dataset = phishingDataset(csv_file)

#split data 80/20
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
#randomize (csv is not randomized)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#load data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("1")

#load model
vocab_size = len(dataset.vocab)
model = MultiInputModel(vocab_size=vocab_size)

# "binary cross-Entropy loss" for binary classification
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("2")


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for specto_img, mfcc_img, transcript, labels in train_loader:
        #prevent previous gradients from continuing in the calculation
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(specto_img.float(), mfcc_img.float(), transcript)
        loss = criterion(outputs.squeeze(), labels.float())
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

print("3")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for specto_img, mfcc_img, transcript, labels in test_loader:
        outputs = model(specto_img.float(), mfcc_img.float(), transcript)

        #convert probability to binary
        predicted = (outputs.squeeze() > 0.5).float() 
        total += labels.size(0)
        correct += (predicted == labels.float()).sum().item()

time.sleep(3)

sys.stdout.flush()

f = open('test.txt', 'w')

f.write(f'Accuracy on test set: {100 * correct / total:.2f}%')

print(f'Accuracy on test set: {100 * correct / total:.2f}%', flush=True)
f.close()