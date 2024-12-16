import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct  # For reading IDX files

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to load MNIST IDX files
def read_idx(file_path):
    with open(file_path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))  # Header format
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return data


def read_labels(file_path):
    with open(file_path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))  # Header format
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


# Custom Dataset to include MNIST and operator images
class CustomDataset(Dataset):
    def __init__(self, mnist_images, mnist_labels, template_dir, transform=None):
        self.mnist_images = mnist_images
        self.mnist_labels = mnist_labels
        self.template_dir = template_dir
        self.transform = transform
        self.operator_images, self.operator_labels = self.load_operator_images()

    def load_operator_images(self):
        operator_images, labels = [], []
        operator_map = {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "plus": 10,
            "minus": 11,
            "times": 12,
            "div": 13,
        }
        for file in os.listdir(self.template_dir):
            if file.endswith(".png"):
                file_key = file.split(".")[0]  # Extract the name without extension
                if file_key in operator_map:
                    img = cv2.imread(
                        os.path.join(self.template_dir, file), cv2.IMREAD_GRAYSCALE
                    )
                    img_resized = cv2.resize(img, (28, 28))
                    operator_images.append(img_resized)
                    labels.append(operator_map[file_key])
                else:
                    print(f"Warning: Unrecognized file '{file}' skipped.")
        return operator_images, labels

    def __len__(self):
        return len(self.mnist_images) + len(self.operator_images)

    def __getitem__(self, idx):
        if idx < len(self.mnist_images):
            img = self.mnist_images[idx]
            label = self.mnist_labels[idx]
            # Resize MNIST images to 32x32
            img_resized = cv2.resize(img, (28, 28))
            img_resized = np.expand_dims(img_resized, axis=0)  # Add channel dimension
            img = torch.tensor(img_resized, dtype=torch.float32) / 255.0
        else:
            img = self.operator_images[idx - len(self.mnist_images)]
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
            label = self.operator_labels[idx - len(self.mnist_images)]
        return img, label


# Data transformations
transform = None  # Already resizing to 32x32 manually

# Change to correct MNIST path
MNIST_PATH = "D:/Coding/ResearchAssignment/SSTestProject/data/MNIST_dataset"
train_images = read_idx(os.path.join(MNIST_PATH, "train-images.idx3-ubyte"))
train_labels = read_labels(os.path.join(MNIST_PATH, "train-labels.idx1-ubyte"))

# Change to correct images directory
TEMPLATE_DIR = "D:/Coding/ResearchAssignment/SSTestProject/data/images"
custom_dataset = CustomDataset(train_images, train_labels, TEMPLATE_DIR, transform)
custom_loader = DataLoader(custom_dataset, batch_size=64, shuffle=True)


# LeNet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 14)  # 10 digits + 4 operators

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # print(f"Shape after conv layers: {x.shape}")  # Debugging
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize model, loss function, and optimizer
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
# use adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in custom_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(custom_loader):.4f}"
    )

# Save the trained model
torch.save(model.state_dict(), "lenet_model.pth")

# Load trained model
model = LeNet().to(device)
model.load_state_dict(torch.load("lenet_model.pth"))
model.eval()


# Function to predict digits/operators from ROI
def predict_digit_or_operator(roi, model):
    roi_resized = cv2.resize(roi, (28, 28))
    roi_tensor = (
        torch.tensor(roi_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    )
    roi_tensor = roi_tensor.to(device)
    with torch.no_grad():
        output = model(roi_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()


# Process video and compute result
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    recognized_sequence = []

    # Track last prediction to avoid duplicates
    last_prediction = None
    # Process every 5th frame
    frame_skip = 5

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to reduce processing frequency
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract Regions of Interest (assume bounding boxes are pre-defined or automated)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter noise
            if w > 10 and h > 10:
                roi = gray[y : y + h, x : x + w]
                predicted = predict_digit_or_operator(roi, model)

                # Avoid appending duplicates
                if predicted != last_prediction:
                    recognized_sequence.append(predicted)
                    last_prediction = predicted

    cap.release()

    # Map predictions to symbols and compute result
    operator_map = {10: "+", 11: "-", 12: "*", 13: "/"}
    expression = "".join(operator_map.get(x, str(x)) for x in recognized_sequence)
    print(f"Recognized Expression: {expression}")
    try:
        result = eval(expression)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error in evaluating expression: {e}")


# Change to your video path
VIDEO_PATH = "D:/Coding/ResearchAssignment/SSTestProject/data/video/video_2.mp4"
process_video(VIDEO_PATH, model)
