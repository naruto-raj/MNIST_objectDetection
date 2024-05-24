import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
import random
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.patches as patches

class MNISTSyntheticDataset(Dataset):
    def __init__(self, data_dir='./data', train=True):
        """
        Initialize the synthetic dataset with CIFAR-10 images embedded randomly in a 224x224 canvas.

        Parameters:
        - data_dir: Directory to download/store CIFAR10 data.
        - train: Boolean indicating whether to load training or testing set
        """
        self.data_dir = data_dir
        self.train = train

        self.image_size = (28, 28)  # CIFAR-10 image dimensions
        self.padding_size = (224, 224)  # Canvas size

        # Load CIFAR-10 data
        self.data = datasets.MNIST(root=self.data_dir, train=self.train, download=True, transform=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        # Randomly resize the image by a factor of 1 to 5
        resize_factor = random.randint(1, 5)
        new_size = (int(self.image_size[0] * resize_factor), int(self.image_size[1] * resize_factor))
        image = image.resize(new_size, Image.LANCZOS)

        # Convert back to numpy array after resizing
        image = np.array(image)
        
        # Create a new blank canvas
        new_image = np.zeros((self.padding_size[0], self.padding_size[1], 1), dtype=np.uint8)

        # Generate random position for the resized image within the canvas
        max_x = self.padding_size[0] - new_size[0]
        max_y = self.padding_size[1] - new_size[1]
        start_x = random.randint(0, max_x) if max_x > 0 else 0
        start_y = random.randint(0, max_y) if max_y > 0 else 0
        end_x = start_x + new_size[0]
        end_y = start_y + new_size[1]

        # Place the CIFAR-10 image at the random position on the new canvas
        new_image[start_x:end_x, start_y:end_y, :] = np.expand_dims(image, axis=-1)

        # Define bounding box and normalize by canvas size [x_min, y_min, x_max, y_max]
        bbox = [
            start_y / self.padding_size[1],  # x_min
            start_x / self.padding_size[0],  # y_min
            end_y / self.padding_size[1],    # x_max
            end_x / self.padding_size[0]     # y_max
        ]
        # Convert new_image, label, bbox to tensors
        new_image_tensor = torch.from_numpy(new_image).permute(2, 0, 1).float()
        label = torch.tensor(label, dtype=torch.int64)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        return new_image_tensor, label, bbox

class DetectionLoss(nn.Module):
    def __init__(self, num_classes):
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.class_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.SmoothL1Loss()

    def forward(self, class_preds, bbox_preds, class_targets, bbox_targets):
        # Calculate the classification loss
        classification_loss = self.class_loss(class_preds, class_targets)

        # Calculate the bounding box loss only for objects (not background)
        pos_indices = class_targets > 0  # Assuming background class is 0
        predicted_boxes = bbox_preds[pos_indices]
        true_boxes = bbox_targets[pos_indices]

        # Regression loss
        regression_loss = self.bbox_loss(predicted_boxes, true_boxes)

        # Combine the two losses
        total_loss = classification_loss + regression_loss
        return classification_loss, regression_loss, total_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # Do not reduce in constructor

    def forward(self, inputs, targets):
        logp = self.ce_loss(inputs, targets)
        p = torch.exp(-logp)
        loss = self.alpha * (1 - p) ** self.gamma * logp
        return loss.mean()

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, preds, targets):
        # Implementation of a simple IoU Loss
        inter = torch.min(preds, targets).prod(dim=1)
        union = torch.max(preds, targets).prod(dim=1)
        iou = inter / union
        return 1 - iou.mean()  # 1 - IoU to make it a minimization problem

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, use_focal=True):
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.class_loss = FocalLoss() if use_focal else nn.CrossEntropyLoss()
        self.bbox_loss = IoULoss()

    def forward(self, class_preds, bbox_preds, class_targets, bbox_targets):
        classification_loss = self.class_loss(class_preds, class_targets)
        pos_indices = class_targets > 0  # Assuming background class is 0
        predicted_boxes = bbox_preds[pos_indices]
        true_boxes = bbox_targets[pos_indices]
        regression_loss = self.bbox_loss(predicted_boxes, true_boxes)
        total_loss = classification_loss + regression_loss
        return classification_loss ,regression_loss, total_loss

def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).float().sum()
    return correct / y_true.shape[0]

def calculate_iou(pred_boxes, target_boxes):
    # Assumes boxes are in [x1, y1, x2, y2] format
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    
    union_area = pred_area + target_area - inter_area
    iou = inter_area / union_area
    return iou.mean()  # Return average IoU for the batch

# Visualisation
def visualize_predictions(model, data_loader, num_images=5, epoch=1, save_path='logs'):
    model.eval()
    save_path = os.path.join(save_path, 'visualizations')
    positive_images = []
    negative_images = []
    
    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        for images, labels, bboxes in data_loader:
            if len(positive_images) >= num_images and len(negative_images) >= num_images:
                break
            images = images.to('cuda')
            class_preds, bbox_preds = model(images)
            images = images.cpu()
            bbox_preds = bbox_preds.cpu()
            labels = labels.cpu()

            # Process each prediction in the batch
            for i in range(len(images)):
                img = images[i].squeeze(0)
                bbox = bbox_preds[i]
                label = labels[i].item()
                pred_label = torch.argmax(class_preds[i]).item()
                
                if pred_label == label:
                    if len(positive_images) < num_images:
                        positive_images.append((img, bbox, label, 'Positive', pred_label))
                else:
                    if len(negative_images) < num_images:
                        negative_images.append((img, bbox, label, 'Negative', pred_label))

    # Now plot positive and negative images
    if positive_images:
        plot_images(positive_images, epoch, save_path, 'positive')
    if negative_images:
        plot_images(negative_images, epoch, save_path, 'negative')


def plot_images(image_data, epoch, save_path, prefix):
    fig, axs = plt.subplots(1, len(image_data), figsize=(20, 4))
    for idx, (img, bbox, label, status, pred_label) in enumerate(image_data):
        ax = axs[idx] if len(image_data) > 1 else axs
        title = f"{status} Sample\nTrue: {label}, Pred: {pred_label}"
        draw_bounding_box(img, bbox, title, ax)
    plt.savefig(f'{save_path}/epoch_{epoch}_{prefix}_visualizations.png')
    plt.close()

def draw_bounding_box(image, bbox, title, ax):
    """
    Draw a bounding box with label on the image.
    """
    # Normalize pixel values to [0, 1] for plotting
    image = (image - image.min()) / (image.max() - image.min())
    
    # Calculate the bounding box coordinates
    ymin, xmin, ymax, xmax = bbox
    xmin *= image.shape[1]
    xmax *= image.shape[1]
    ymin *= image.shape[0]
    ymax *= image.shape[0]
    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.imshow(image, cmap='gray')
    ax.add_patch(rect)
    ax.set_title(title)
    ax.axis('off')

def plot_samples_with_labels_and_bbox(dataset):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Plot 2*5 samples with their labels and bbox coordinates
    for i in range(2):
        for j in range(5):
            ax = axes[i, j]
            idx = i * 5 + j
            image, label, bbox = dataset[idx]

            # Plot the image
            ax.imshow(image.squeeze(), cmap='gray')
            ax.axis('off')

            # Add bounding box
            rect = patches.Rectangle((bbox[0] * 224, bbox[1] * 224), bbox[2] * 224 - bbox[0] * 224, bbox[3] * 224 - bbox[1] * 224,
                                     linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

            # Add label
            ax.set_title(f"Label: {label.item()}")

    plt.tight_layout()
    plt.show()

def train_and_validate(model, train_loader, val_loader, optimizer, loss_function, epochs, visualize_after_train=False,log_dir='object_detection'):
    # Initialize TensorBoard Summary Writer
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_classification_loss,train_iou_loss, train_accuracy, train_iou = 0.0, 0.0, 0.0, 0.0
        for images, class_targets, bbox_targets in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{epochs}', leave=False):
            images = images.to(device)
            class_targets = class_targets.to(device)
            bbox_targets = bbox_targets.to(device)
            
            optimizer.zero_grad()
            class_preds, bbox_preds = model(images)
            
            classification_loss, iou_loss,loss = loss_function(class_preds, bbox_preds, class_targets, bbox_targets)
            loss.backward()
            optimizer.step()
            
            train_classification_loss += classification_loss.item()
            train_iou_loss += iou_loss.item()
            
            train_accuracy += calculate_accuracy(class_preds, class_targets.squeeze())
            train_iou += calculate_iou(bbox_preds, bbox_targets.squeeze())

        # After each epoch calculate and log average values
        avg_train_classification_loss = train_classification_loss / len(train_loader)
        avg_train_iou_loss = train_iou_loss / len(train_loader)
        avg_train_accuracy = train_accuracy / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)

        writer.add_scalar('Loss/Avg_Train_classifcation', avg_train_classification_loss, epoch)
        writer.add_scalar('Loss/Avg_Train_iou', avg_train_iou_loss, epoch)
        writer.add_scalar('Accuracy/Avg_Train', avg_train_accuracy, epoch)
        writer.add_scalar('IoU/Avg_Train', avg_train_iou, epoch)
        
        # Validation Phase
        model.eval()
        val_classification_loss,val_iou_loss, val_accuracy, val_iou = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, class_targets, bbox_targets in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{epochs}', leave=False):
                images = images.to(device)
                class_targets = class_targets.to(device)
                bbox_targets = bbox_targets.to(device)

                class_preds, bbox_preds = model(images)
                classification_loss, iou_loss,loss = loss_function(class_preds, bbox_preds, class_targets, bbox_targets)

                val_classification_loss += classification_loss.item()
                val_iou_loss += iou_loss.item()
                
                val_accuracy += calculate_accuracy(class_preds, class_targets.squeeze())
                val_iou += calculate_iou(bbox_preds, bbox_targets.squeeze())

        avg_val_classification_loss = val_classification_loss / len(val_loader)
        avg_val_iou_loss = val_iou_loss / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        writer.add_scalar('Loss/Avg_Validation_classifcation', avg_val_classification_loss, epoch)
        writer.add_scalar('Loss/Avg_Validation_iou', avg_val_iou_loss, epoch)
        writer.add_scalar('Accuracy/Avg_Validation', avg_val_accuracy, epoch)
        writer.add_scalar('IoU/Avg_Validation', avg_val_iou, epoch)
        
        # Optionally visualize predictions on validation data
        if visualize_after_train:
            visualize_predictions(model, val_loader, num_images=5,epoch=epoch+1,save_path=log_dir)

        # Print training and validation summary for the epoch
        print(f'Epoch [{epoch + 1}/{epochs}] - Train classification Loss: {train_classification_loss / len(train_loader):.4f}, Train iou Loss: {train_iou_loss / len(train_loader):.4f}, Accuracy: {train_accuracy / len(train_loader):.4f}, IoU: {train_iou / len(train_loader):.4f}')
        print(f'Epoch [{epoch + 1}/{epochs}] - Val classification Loss: {val_classification_loss / len(val_loader):.4f}, Val iou Loss: {val_iou_loss / len(val_loader):.4f}, Accuracy: {val_accuracy / len(val_loader):.4f}, IoU: {val_iou / len(val_loader):.4f}')
    writer.close()