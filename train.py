import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import phishingDataset 
from new_model import *
import time
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

csv_file = "./data.csv"
batch_size = 32
num_epochs = 10
learning_rate = 0.001

dataset = phishingDataset(csv_file)

#split data 80/20
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
#randomize (csv is not randomized)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#load data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#load model
vocab_size = len(dataset.vocab)
model = MultiInputModel(vocab_size=vocab_size)
torch.save(dataset.vocab, "vocab.pth")
# print("done")

sparse_params = []
dense_params = []

for name, param in model.named_parameters():
    if 'embedding' in name:
        sparse_params.append(param)
    else:
        dense_params.append(param)

# Use SparseAdam to handle sparse gradients (custom model uses sparse gradients)
optimizer = optim.Adam(dense_params, lr=learning_rate)
sparse_optimizer = optim.SparseAdam(sparse_params, lr=learning_rate)

# "binary cross-Entropy loss" for binary classification
criterion = nn.BCELoss()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for specto_img, mfcc_img, transcript, labels in train_loader:
        #Prevent previous gradients from continuing in the calculation
        optimizer.zero_grad()
        sparse_optimizer.zero_grad()
        
        # forward pass
        outputs = model(specto_img.float(), mfcc_img.float(), transcript)
        loss = criterion(outputs.squeeze(), labels.float())
        
        loss.backward()
        optimizer.step()
        sparse_optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Save the model
# torch.save(model.state_dict(), 'multi_input_model_2.pth')

model.eval()
all_labels = []
all_predictions = []
correct = 0
total = 0

with torch.no_grad():
    for specto_img, mfcc_img, transcript, labels in test_loader:
        outputs = model(specto_img.float(), mfcc_img.float(), transcript)

        #convert probability to binary
        predicted = (outputs.squeeze() > 0.5).float() 
        all_labels.extend(labels.tolist())
        all_predictions.extend(predicted.tolist())
        
        total += labels.size(0)
        correct += (predicted == labels.float()).sum().item()

# Calculate and pront metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)
conf_matrix = confusion_matrix(all_labels, all_predictions)

print(f"Accuracy (sklearn): {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")