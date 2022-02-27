import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt

# importing dependencies related to image transformations
import torchvision
from torchvision import transforms
from PIL import Image

# importing dependencies related to data loading
from torchvision import datasets
from torch.utils.data import DataLoader

image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


# loading data
data = {
    'train':
    datasets.ImageFolder(root='./dataset/train',
                         transform=image_transforms['train']),
    'test':
    datasets.ImageFolder(root='./dataset/test',
                         transform=image_transforms['test']),
}

# Dataloader iterators, used for making batches
dataloaders = {
    'train': DataLoader(data['train'], batch_size=100, shuffle=True),
    'test': DataLoader(data['test'], batch_size=100, shuffle=True)}


# In[5]:


# loading MobileNetv2
model = torch.hub.load('pytorch/vision:v0.6.0',
                       'mobilenet_v2', pretrained=True)


# In[6]:


# freezing the initial layers of MobileNetv2
for param in model.parameters():
    param.requires_grad = False


# In[7]:


# adding our own classifier
model.classifier[1] = nn.Sequential(
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(32, 2),
    nn.LogSoftmax(dim=1))


# In[8]:


# checking trainable parameters
print(sum(p.numel() for p in model.parameters() if p.requires_grad))


# In[17]:


# training data

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in (train_loader):
            optimizer.zero_grad()
            inputs, targets = batch
            #inputs = inputs.to(device)
            #targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in (val_loader):
            inputs, targets = batch
            #inputs = inputs.to(device)
            output = model(inputs)
            #targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[
                               1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(
            epoch, training_loss, valid_loss, num_correct / num_examples))

    return model


# In[22]:


# testing data
def test_model(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloaders['test']:
            #images, labels = data[0].to(device), data[1].to(device)
            images, labels = data[0], data[1]
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('correct: {:d}  total: {:d}'.format(correct, total))
    print('accuracy = {:f}'.format(correct / total))


optimizer = optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()


# train the model first 20 epochs
model = train(model, optimizer, loss,
              dataloaders['train'], dataloaders['test'], 10)
torch.save(model, "model.pth")
