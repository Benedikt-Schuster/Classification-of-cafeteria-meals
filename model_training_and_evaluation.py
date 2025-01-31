import os
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import random_split, DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import time
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from torchprofile import profile_macs
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForImageClassification
from transformers import EfficientNetForImageClassification

class CanteenFoodDataset(Dataset):
    def __init__(self, root_dir):
        '''
        Custom dataset for canteen food images, that automatically loads images and their corresponding label from a folder structure.
        The folder structure should be as follows (labels are automatically named after sub_folders):

        root_dir
        ├── class_1
        │   ├── image_1.jpg
        │   ├── image_2.jpg
        │   └── ...
        ├── class_2

        Args:
            root_dir (string): Directory with all the images.
        '''
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self._load_images_from_folders()

    def _load_images_from_folders(self):
          for class_idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
              class_path = os.path.join(self.root_dir, class_name)
              if os.path.isdir(class_path):
                  self.class_to_idx[class_name] = class_idx
                  self.idx_to_class[class_idx] = class_name
                  for file_name in os.listdir(class_path):
                      file_path = os.path.join(class_path, file_name)
                      if os.path.isfile(file_path):
                          self.image_paths.append(file_path)
                          self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        return image, label

class TransformableSubset(Dataset):
    def __init__(self, subset, transform):
        '''
        Custom class to allow assigning transforms to a dataset after a random_split
        '''
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# Split into train, val and test
dataset = CanteenFoodDataset('.venv/Bilder') # <-- put your img_directory here
train_val, test = random_split(dataset, [0.8, 0.2])
train, val = random_split(train_val, [0.8, 0.2])
train_dataset = TransformableSubset(train, transform['train'])
val_dataset = TransformableSubset(val, transform['val'])
test_dataset = TransformableSubset(test, transform['val'])

NUM_CLASSES = len(dataset.class_to_idx)
BATCH_SIZE = 16 # Should probably be adjusted to fit problem
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create data loaders.
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

def train_model(net, optimizer, scheduler, early_stopping, loss_fct, num_epochs, best_model_path):

    net = net.to(DEVICE)

    train_losses = []
    val_losses = []
    learning_rates = []

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        # Training
        net.train()
        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = net(X_batch)["logits"]

            loss = loss_fct(y_pred, y_batch)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_dataloader)

        # Validation
        net.eval()
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_dataloader:
                X_val_batch, y_val_batch = X_val_batch.to(DEVICE), y_val_batch.to(DEVICE)

                y_pred_val = net(X_val_batch)["logits"]
                loss_val = loss_fct(y_pred_val, y_val_batch)
                val_losses.append(loss_val.item())

                epoch_val_loss += loss_val.item()

        epoch_val_loss /= len(val_dataloader)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        scheduler.step(epoch_val_loss)

        # Saving the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(net.state_dict(), best_model_path)
            print(f"Best model saved with val loss: {best_val_loss:.5f}")

        if early_stopping(epoch_val_loss):
            print("Early stopping triggered.")
            break

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_train_loss:.5f} --- Val Loss: {epoch_val_loss:.5f} --- LR: {learning_rates[epoch]}")

    return train_losses, val_losses, learning_rates

def print_graph(train_losses, val_losses, learning_rates):

    # Calculate total number of data points for x-axis
    total_train_data_points = len(train_losses)
    total_val_data_points = len(val_losses)
    epochs = len(learning_rates)

    # Create x-axis values for train and validation losses
    x_train = np.linspace(1, epochs, total_train_data_points) # Stretch train losses across epochs
    x_val = np.linspace(1, epochs, total_val_data_points) # Stretch val losses across epochs
    x_lr = np.linspace(1, len(learning_rates), len(learning_rates))


    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot train and validation losses on the first y-axis
    ax1.plot(x_train, train_losses, label="Train Loss", color="blue")
    ax1.plot(x_val, val_losses, label="Validation Loss", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    # Create a second y-axis for learning rates
    ax2 = ax1.twinx()
    ax2.plot(x_lr, learning_rates, label="Learning Rate", color="green")
    ax2.set_ylabel("Learning Rate", color="green")
    ax2.tick_params(axis='y', labelcolor="green")

    # Add title, legend, and grid
    plt.title("Training and Validation Loss per Epoch with Learning Rate")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    ax1.grid()

    plt.show()

def find_best_lr(net, optimizer, loss_fct):

    lrs = []
    losses = []
    lr_multiplier = 1.1  # Factor to increase learning rate each step
    current_lr = 1e-7
    repeated_data_loader = cycle(train_dataloader)

    while(current_lr <= 0.01):
        inputs, targets = next(repeated_data_loader)
        optimizer.param_groups[0]['lr'] = current_lr  # Update the learning rate
        lrs.append(current_lr)

        # Forward pass
        outputs = net(inputs)['logits']
        loss = loss_fct(outputs, targets)
        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Increase the learning rate
        current_lr *= lr_multiplier

    # Plot the learning rate vs loss
    plt.plot(lrs, losses)
    plt.xscale('log')  # Use log scale for learning rate
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()

def evaluate_model(net):

    net.eval()
    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for data, target in test_dataloader:
            start_time = time.time()
            print(data)
            print(data.shape)
            output = net(data)["logits"]
            print(output)
            _, preds = torch.max(output, 1)
            end_time = time.time()

            inference_time = end_time - start_time
            inference_times.append(inference_time)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Average Inference Time per Sample: {sum(inference_times) / len(inference_times):.6f} seconds')
    print(f'Min Inference Time: {min(inference_times):.6f} seconds')
    print(f'Max Inference Time: {max(inference_times):.6f} seconds')


    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.idx_to_class.values())
    disp.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=90)
    plt.show()

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    test_iter = iter(test_dataloader)
    X_batch, y_batch = next(test_iter)
    input_tensor = X_batch.to(DEVICE)
    macs = profile_macs(net, input_tensor)
    flops = 2 * macs  # MACs * 2 = FLOPs
    total_params = sum(p.numel() for p in net.parameters())
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Parameter: {total_params / 1e6:.2f} Millionen")


'CNN trained on own data'
class Net(nn.Module):
    def __init__(self, num_classes = 10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features= 64 * 56 * 56, out_features= 128)
        self.fc2 = nn.Linear(in_features= 128, out_features=NUM_CLASSES)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return {"logits": x}

net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.00020)
loss_fct = nn.CrossEntropyLoss()
find_best_lr(net, optimizer, loss_fct)

net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.00030)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)
early_stopping = EarlyStopping(patience=4, delta=0.001)
loss_fct = nn.CrossEntropyLoss()

num_epochs = 50

train_losses, val_losses, learning_rates = train_model(net, optimizer, scheduler, early_stopping, loss_fct, num_epochs, '/content/drive/MyDrive/Models/ownNet')
print_graph(train_losses, val_losses, learning_rates)

evaluate_model(net)

#Saving the model
model_save_path = "net_model.pth"
torch.save(net.state_dict(), model_save_path)
print(f"Modell gespeichert unter: {model_save_path}")


'Transfer Learning on EfficientNet Model trained with Food101 Dataset'

efficient_net_food101 = AutoModelForImageClassification.from_pretrained("gabrielganan/efficientnet_b1-food101")

#Freeze layers
for param in efficient_net_food101.parameters():
    param.requires_grad = False

#Modify classifier
efficient_net_food101.classifier = nn.Linear(in_features = efficient_net_food101.classifier.in_features, out_features = NUM_CLASSES)

optimizer = torch.optim.Adam(efficient_net_food101.classifier.parameters(), lr=0.002) #Only train classifier layer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)
early_stopping = EarlyStopping(patience=5, delta=0.001)
loss_fct = nn.CrossEntropyLoss()

find_best_lr(efficient_net_food101, optimizer, loss_fct)

num_epochs = 50

train_losses, val_losses, learning_rates = train_model(efficient_net_food101, optimizer, scheduler, early_stopping, loss_fct, num_epochs, '/content/drive/MyDrive/Models/efficient_net_food101')
print_graph(train_losses, val_losses, learning_rates)

evaluate_model(efficient_net_food101)

#Saving the model
model_save_path = "efficient_net_food101.pth"
torch.save(net.state_dict(), model_save_path)
print(f"Modell gespeichert unter: {model_save_path}")


'Transfer Learning on EfficientNet Model trained without Food101 Dataset'

efficient_net = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b1")

#Freeze layers
for param in efficient_net.parameters():
    param.requires_grad = False

#Modify classifier
efficient_net.classifier = nn.Linear(in_features = efficient_net.classifier.in_features, out_features = NUM_CLASSES)

optimizer = torch.optim.Adam(efficient_net.classifier.parameters(), lr=0.002) #Only train classifier layer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=2)
early_stopping = EarlyStopping(patience=5, delta=0.001)
loss_fct = nn.CrossEntropyLoss()

find_best_lr(efficient_net, optimizer, loss_fct)
num_epochs = 50

train_losses, val_losses, learning_rates = train_model(efficient_net, optimizer, scheduler, early_stopping, loss_fct, num_epochs, '/content/drive/MyDrive/Models/efficient_net')
print_graph(train_losses, val_losses, learning_rates)

efficient_net = torch.load(".venv/Model/efficient_net")

evaluate_model(efficient_net)

#Saving the model
model_save_path = "efficient_net.pth"
torch.save(net.state_dict(), model_save_path)
print(f"Modell gespeichert unter: {model_save_path}")