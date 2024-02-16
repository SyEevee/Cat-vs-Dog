# CNN for cat vs dog dataset

# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Setting the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setting the hyperparameters
in_channels = 3
num_classes = 2
learning_rate = 0.001
batch_size = 32
num_epochs = 1

# Creating the model
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
model.aux_logits=False
n_inputs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(n_inputs, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.Dropout(0.5),
    nn.Linear(32, num_classes)
)
model.to(device)

# Loading the data
train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
])

test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
])

train_dataset = datasets.ImageFolder(root='dataset/training_set',
                                        transform=train_transform,)
test_dataset = datasets.ImageFolder(root='dataset/test_set',
                                        transform=test_transform)

train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# using lr scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training the model
for epoch in range(num_epochs):
    losses = []
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        scheduler.step()

        print(f'Epoch: {epoch+1}/{num_epochs}, Step: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {scores.argmax(1).eq(targets).sum().item()/batch_size*100:.4f}%', end='\r')
    print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {np.mean(losses):.4f}', end='\n')

# Checking accuracy on training & test set
def check_accuracy(loader, model, train):
    if train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

check_accuracy(train_loader, model, train=True)
check_accuracy(test_loader, model, train=False)

# Saving the model
torch.save(model, 'cats_vs_dogs.pth')