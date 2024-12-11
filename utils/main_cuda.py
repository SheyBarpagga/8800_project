import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils.model import MultiInputModel
from dataset import phishingDataset 
import sys
import time
import torch.cuda

import torchtext; torchtext.disable_torchtext_deprecation_warning()

csv_file = "./data.csv"
batch_size = 32
num_epochs = 10
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

dataset = phishingDataset(csv_file)

#split data 80/20
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
#randomize (csv is not randomized)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#load data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#load model and move to device
vocab_size = len(dataset.vocab)
model = MultiInputModel(vocab_size=vocab_size).to(device)

# "binary cross-Entropy loss" for binary classification
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for specto_img, mfcc_img, transcript, labels in train_loader:
        # move data to device
        specto_img, mfcc_img, transcript, labels = (
            specto_img.float().to(device), 
            mfcc_img.float().to(device), 
            transcript.to(device), 
            labels.float().to(device)
        )

        #prevent previous gradients from continuing in the calculation
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(specto_img, mfcc_img, transcript)
        loss = criterion(outputs.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for specto_img, mfcc_img, transcript, labels in test_loader:
        specto_img, mfcc_img, transcript, labels = (
            specto_img.float().to(device), 
            mfcc_img.float().to(device), 
            transcript.to(device), 
            labels.float().to(device)
        )

        outputs = model(specto_img, mfcc_img, transcript)
        predicted = (outputs.squeeze() > 0.5).float() 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

time.sleep(3)

sys.stdout.flush()

with open('test.txt', 'w') as f:
    f.write(f'Accuracy on test set: {100 * correct / total:.2f}%')
f.close()

print(f'Accuracy on test set: {100 * correct / total:.2f}%', flush=True)
