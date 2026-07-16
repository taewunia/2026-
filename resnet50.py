import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import timm
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import F1Score, Accuracy, MetricCollection
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

BATCH_SIZE = 16
EPOCH = 10
LR = 0.0001

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

metrics = MetricCollection({
    'f1_score': F1Score(task="multiclass", num_classes=10, average='macro'),
    'acc': Accuracy(task="multiclass", num_classes=10)
})

metrics = metrics.to(device)

imagenette_classes = [
    "Tench",
    "English Springer",
    "Cassette Player",
    "Chain Saw",
    "Church",
    "French Horn",
    "Garbage Truck",
    "Gas Pump",
    "Golf Ball"
    "Parachute"
]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_DS = torchvision.datasets.Imagenette(root='./data', split='train', size='320px', download=True,
                                           transform=train_transform)
val_DS = torchvision.datasets.Imagenette(root='./data', split='val', size='320px', download=True,
                                         transform=val_transform)

train_DL = torch.utils.data.DataLoader(dataset=train_DS, shuffle=True, batch_size=BATCH_SIZE)
val_DL = torch.utils.data.DataLoader(dataset=val_DS, shuffle=False, batch_size=BATCH_SIZE)


class Res_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(model_name='resnet50', pretrained=True, num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return x


def visualize_feature_map(model, DL, device):
    model.eval()
    with torch.no_grad():
        sample, labels = next(iter(DL))
        img_tensor = sample[0].unsqueeze(0).to(device)
        label = labels[0].item()

        classes = [
            "Tench", "English Springer", "Cassette Player", "Chain Saw", "Church",
            "French Horn", "Garbage Truck", "Gas Pump", "Golf Ball", "Parachute"
        ]

        m = model.model
        x = m.conv1(img_tensor)
        x = m.bn1(x)
        x = m.act1(x)
        x = m.maxpool(x)
        output = m.layer1(x)

        output = output.squeeze(0).cpu().numpy()

        fig = plt.figure(figsize=(12, 10))
        grid = plt.GridSpec(5, 4, hspace=0.4, wspace=0.3)

        ax_main = fig.add_subplot(grid[0, :])
        origin_img = sample[0].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        origin_img = origin_img * std + mean
        ax_main.imshow(origin_img.clip(0, 1))
        ax_main.set_title(f"Original Image (Label: {classes[label]})", fontsize=14, fontweight='bold')
        ax_main.axis('off')

        for i in range(16):
            ax_feat = fig.add_subplot(grid[i // 4 + 1, i % 4])
            ax_feat.imshow(output[i], cmap='viridis')
            ax_feat.set_title(f"Channel {i}", fontsize=10)
            ax_feat.axis('off')

        plt.show()


def visualize_gradcam(model, DL, device):
    model.eval()
    target_layers = [model.model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    dataiter = iter(DL)
    images, labels = next(dataiter)
    input_tensor = images[0].unsqueeze(0).to(device)

    targets = [ClassifierOutputTarget(labels[0].item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    img_numpy = images[0].permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_original = std * img_numpy + mean
    img_original = np.clip(img_original, 0, 1)

    visualization = show_cam_on_image(img_original, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.axis('off')

    plt.show()


model = Res_model()
model.to(device)
print(model)

visualize_feature_map(model=model, DL=val_DL, device=device)
train_history = []
val_history = []

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCH):
    model.train()
    train_bar = tqdm(train_DL, unit='batch', desc=f"Training {epoch + 1}/{EPOCH}", colour='green')
    avg_train_loss = 0
    avg_val_loss = 0
    total_train_loss = 0
    total_val_loss = 0
    for x, y in train_bar:
        x, y = x.to(device), y.to(device)
        result = model(x)
        loss = criterion(result, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        tqdm.set_postfix(train_bar, loss=loss.item())
    avg_train_loss = total_train_loss / len(train_DL)
    train_history.append(avg_train_loss)

    val_bar = tqdm(val_DL, unit='batch', desc=f"Training {epoch + 1}/{EPOCH}", colour='red')
    model.eval()

    with torch.no_grad():
        for x, y in val_bar:
            x, y = x.to(device), y.to(device)
            result = model(x)
            loss = criterion(result, y)
            total_val_loss += loss.item()
            tqdm.set_postfix(val_bar, loss=loss.item())
            metrics.update(result, y)
        avg_val_loss = total_val_loss / len(val_DL)
        val_history.append(avg_val_loss)
        test_result = metrics.compute()

    tqdm.write(f"avg_train_loss:{avg_train_loss:.3f} avg_val_loss:{avg_val_loss:.3f}")
    tqdm.write(f"Accuracy:{test_result['acc'].item():.3f} F1_score:{test_result['f1_score'].item():.3f}")

    visualize_feature_map(model=model, DL=val_DL, device=device)
    visualize_gradcam(model=model, DL=val_DL, device=device)

    metrics.reset()

plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCH + 1), train_history, label='Train Loss', color='green', marker='o')
plt.plot(range(1, EPOCH + 1), val_history, label='Val Loss', color='red', marker='s')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.xticks(range(1, EPOCH + 1))
plt.legend()
plt.grid(True)
plt.show()