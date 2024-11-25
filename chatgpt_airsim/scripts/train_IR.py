import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Define the U-Net Model
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        
        # Downsampling
        self.enc1 = self.contracting_block(1, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.contracting_block(512, 1024)
        
        # Upsampling
        self.up1 = self.expansive_block(1024, 512)
        self.up2 = self.expansive_block(512, 256)
        self.up3 = self.expansive_block(256, 128)
        self.up4 = self.expansive_block(128, 64)
        
        # Output
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        return block
    
    def expansive_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block
        
    def forward(self, x):
         # Downsampling
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
            
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
            
        # Upsampling with concatenation
        up1 = self.up1(bottleneck)
        up2 = self.up2(up1 + enc4)
        up3 = self.up3(up2 + enc3)
        up4 = self.up4(up3 + enc2)
        
        # Final output and upsample to the original size # CHANGED FROM 128 TO 256
        out = self.final(up4 + enc1)
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=False)  


        return out


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(weight=self.alpha)(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = ((1 - pt) ** self.gamma) * BCE_loss
        return F_loss

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0005):  # Increased patience, refined min_delta
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Custom Dataset class to load thermal IR images and segmentation maps
class ThermalSegmentationDataset(nn.Module):
    def __init__(self, image_dir, label_dir, transform=None, label_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx >= len(self.images) or idx >= len(self.labels):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.images)}")

        image_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])

        image = Image.open(image_path).convert("L")
        label = Image.open(label_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        label = np.array(label, dtype=np.int64)
        label = remap_labels(label)
        label = torch.from_numpy(label)

        return image, label

           

# Function to remap labels
def remap_labels(label):
    remap_dict = {
        0: 0,      # Background
        72: 1,     # Class 1
        100: 2,    # Class 2
        109: 3,    # Class 3
        115: 4,    # Fire (Class 4)
        144: 5,    # Class 5
    }
    
    remapped_label = np.copy(label)
    for old_value, new_value in remap_dict.items():
        remapped_label[label == old_value] = new_value
    remapped_label[~np.isin(label, list(remap_dict.keys()))] = 0
    return remapped_label

# Function to calculate pixel-wise accuracy
def pixel_accuracy(output, label):
    _, preds = torch.max(output, 1)
    correct = (preds == label).sum().item()
    total = torch.numel(label)
    return correct / total

# Function to calculate IoU for a specific class
def calculate_iou(preds, labels, num_classes):
    iou_list = []
    _, predicted = torch.max(preds, 1)
    
    for cls in range(num_classes):
        intersection = ((predicted == cls) & (labels == cls)).sum().item()
        union = ((predicted == cls) | (labels == cls)).sum().item()
        if union == 0:
            iou_list.append(np.nan)  # Avoid division by zero
        else:
            iou_list.append(intersection / union)
    
    return np.nanmean(iou_list)  # Return mean IoU for all classes

# Helper function to calculate precision, recall, and F1 score for each class
def calculate_metrics(preds, labels, num_classes):
    _, predicted = torch.max(preds, 1)  # Get the predicted class
    
    precision_list = []
    recall_list = []
    f1_list = []
    
    for cls in range(num_classes):
        # Convert to binary masks for the current class (cls)
        preds_cls = (predicted == cls).cpu().numpy().flatten()
        labels_cls = (labels == cls).cpu().numpy().flatten()

        if np.sum(labels_cls) == 0 and np.sum(preds_cls) == 0:
            # Skip calculation for empty classes in both prediction and ground truth
            continue
        
        # Calculate precision, recall, and F1 for the current class
        precision = precision_score(labels_cls, preds_cls, zero_division=1)
        recall = recall_score(labels_cls, preds_cls, zero_division=1)
        f1 = f1_score(labels_cls, preds_cls, zero_division=1)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    # Return the mean precision, recall, and F1 across all classes
    return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)
 


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths
image_dir = r'IRNoisyImages'
label_dir = r'SegmentationImages'

# INCREASED IMAGE RESOLUTION FROM 128x128 to 256x256 + ADDED MORE AUGMENTATION TECHNIQUES
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Label transform to resize the labels as well
label_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=Image.NEAREST),
])

# Create dataset
dataset = ThermalSegmentationDataset(image_dir, label_dir, transform=transform, label_transform=label_transform)

# Split the dataset into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation
batch_size = 22 # unet2 model -> 3
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create U-Net model instance
model = UNet(num_classes=6).to(device)


# Define class weights (higher weight for fire class - class 4)
class_weights = torch.tensor([1.0, 2.0, 2.0, 2.0, 5.0, 2.0]).to(device)

# Define the loss function with class weights # CHANGED LOSS
criterion = nn.CrossEntropyLoss(weight=class_weights) #FocalLoss(alpha=class_weights)

# Define the optimizer with weight decay #REDUCED THE LEARNING RATE FOR MORE GRADUAL LEARNING + WEIGHT DECAY TO IMPROVE REGULARIZATION 
optimizer = optim.Adam(model.parameters(), lr=5.611133710685557e-05, weight_decay=1.2730567599855712e-05) #unet2 model -> lr=1e-4, weight_decay=1e-4
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


model.to(device)  # Move model to GPU if available

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Tracking metrics
train_losses, val_losses, iou_scores, accuracies, precisions, recalls, f1_scores = [], [], [], [], [], [], []

# Early stopping setup #INCREASED PATIENCE TO ALLOW MODEL MORE TIME TO RECOVER FROM PLATEAUS
early_stopping = EarlyStopping(patience=15, min_delta=0.0005)


# Training loop
# num_epochs = 13 # unet2 model -> 100
# total_start_time = time.time()

# for epoch in range(num_epochs):
#     epoch_start_time = time.time()
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images) 
#         loss = criterion(outputs, labels.long())
#         loss.backward()

#         # Gradient clipping
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

#         optimizer.step()
#         running_loss += loss.item()
    
       

#     train_losses.append(running_loss / len(train_loader))  # Save training loss
    
#     # Validation phase
#     model.eval()
#     val_loss = 0.0
#     total_iou = 0.0
#     total_acc = 0.0
#     total_precision = 0.0
#     total_recall = 0.0
#     total_f1 = 0.0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)#['out']
#             loss = criterion(outputs, labels.long())
#             val_loss += loss.item()
            
#             # Calculate accuracy and IoU
#             total_acc += pixel_accuracy(outputs, labels)
#             total_iou += calculate_iou(outputs, labels, num_classes=6)

#             precision, recall, f1 = calculate_metrics(outputs, labels, num_classes=6)
#             total_precision += precision
#             total_recall += recall
#             total_f1 += f1
    
#     val_losses.append(val_loss / len(val_loader))  # Save validation loss
#     accuracies.append(total_acc / len(val_loader))  # Save pixel accuracy
#     iou_scores.append(total_iou / len(val_loader))  # Save IoU
#     precisions.append(total_precision / len(val_loader))  # Save Precision
#     recalls.append(total_recall / len(val_loader))  # Save Recall
#     f1_scores.append(total_f1 / len(val_loader))  # Save F1 Score

#     # Learning rate scheduler step
#     scheduler.step(val_losses[-1])

#     # Check early stopping
#     early_stopping(val_losses[-1])
#     if early_stopping.early_stop:
#         print(f"Early stopping triggered at epoch {epoch+1}")
#         break

#     # End timing for the current epoch
#     epoch_end_time = time.time()
#     epoch_duration = epoch_end_time - epoch_start_time

#     print(f"Epoch [{epoch+1}/{num_epochs}], "
#           f"Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, "
#           f"Pixel Accuracy: {accuracies[-1]:.4f}, IoU: {iou_scores[-1]:.4f}, "
#           f"Precision: {precisions[-1]:.4f}, Recall: {recalls[-1]:.4f}, F1 Score: {f1_scores[-1]:.4f}, "
#           f"Epoch Time: {epoch_duration:.2f} seconds")

# # End timing the entire training process
# total_end_time = time.time()
# total_duration = total_end_time - total_start_time

# # Create a dictionary of metrics
# metrics_dict = {
#     'Epoch': list(range(1, len(train_losses) + 1)),
#     'Training Loss': train_losses,
#     'Validation Loss': val_losses,
#     'Pixel Accuracy': accuracies,
#     'IoU': iou_scores,
#     'Precision': precisions,
#     'Recall': recalls,
#     'F1 Score': f1_scores
# }

# # Convert the dictionary to a pandas DataFrame
# metrics_df = pd.DataFrame(metrics_dict)

# # Save the DataFrame to a CSV file
# metrics_df.to_csv('training_metrics_unet3_tuned.csv', index=False)

# print("Metrics saved to training_metrics_unet3_tuned.csv")

# print(f"Total training time: {total_duration:.2f} seconds")

# # Save the trained model
# torch.save(model.state_dict(), 'unet_thermal_model_3_tuned.pth')
# print("Model training complete and saved!")

# Define test dataset paths
test_image_dir = r'TestIRImages'
test_label_dir = r'TestSegmentationImages'

# Create test dataset
test_dataset = ThermalSegmentationDataset(test_image_dir, test_label_dir, transform=transform, label_transform=label_transform)

# Create DataLoader for test dataset
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model on the test set
model.eval()
test_loss = 0.0
total_acc = 0.0
total_iou = 0.0
total_precision = 0.0
total_recall = 0.0
total_f1 = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # Model predictions
        loss = criterion(outputs, labels.long())
        test_loss += loss.item()

        # Calculate metrics
        total_acc += pixel_accuracy(outputs, labels)
        total_iou += calculate_iou(outputs, labels, num_classes=6)
        precision, recall, f1 = calculate_metrics(outputs, labels, num_classes=6)
        total_precision += precision
        total_recall += recall
        total_f1 += f1

# Average metrics across the test set
num_batches = len(test_loader)
test_loss /= num_batches
test_acc = total_acc / num_batches
test_iou = total_iou / num_batches
test_precision = total_precision / num_batches
test_recall = total_recall / num_batches
test_f1 = total_f1 / num_batches

# Print test set metrics
print(f"Test Set Metrics:")
print(f"Loss: {test_loss:.4f}")
print(f"Pixel Accuracy: {test_acc:.4f}")
print(f"IoU: {test_iou:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")
