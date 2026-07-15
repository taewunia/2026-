import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import timm
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import numpy as np

from torchmetrics import F1Score, Accuracy, MetricCollection

BATCH_SIZE = 64
EPOCH = 10
LR = 0.00001

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std = [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std = [0.229, 0.224, 0.225])
])

train_DS = torchvision.datasets.Imagenette(root='./data', split='train', size='320px', download=True, transform=train_transform)
val_DS = torchvision.datasets.Imagenette(root='./data', split='val', size='320px', download=True, transform=val_transform)

train_DL = torch.utils.data.DataLoader(dataset=train_DS, shuffle=True, batch_size=BATCH_SIZE)
val_DL = torch.utils.data.DataLoader(dataset=val_DS, shuffle=False, batch_size=BATCH_SIZE)

class TR_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(model_name='swin_tiny_patch4_window7_224', pretrained=True, num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return x

model = TR_model()
model.to(device)
print(model)

train_history = []
val_history = []

metrics = MetricCollection({
    'f1_score':F1Score(task="multiclass",num_classes=10, average='macro'),
    'acc':Accuracy(task="multiclass",num_classes=10)
})

metrics = metrics.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCH):
    avg_train_loss = 0
    avg_val_loss = 0
    total_train_loss = 0
    total_val_loss = 0
    model.train()
    train_bar = tqdm(train_DL, unit='batch', desc=f"Training {epoch + 1}/{EPOCH}", colour='green')
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


    target_layers = [model.model.layers[-1].blocks[-1].norm2]


    def reshape_transform(tensor):
        if tensor.dim() == 3:
            B, L, C = tensor.shape
            H = int(L ** 0.5)
            W = H
            result = tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            result = tensor.permute(0, 3, 1, 2)
        return result

    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    dataiter = iter(val_DL)
    images, labels = next(dataiter)
    input_tensor = images[0].unsqueeze(0).to(device)

    targets = [ClassifierOutputTarget(labels[0].item())]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    img_numpy = images[0].permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_original = std * img_numpy + mean
    img_original = np.clip(img_original, 0, 1)

    visualization = show_cam_on_image(img_original, grayscale_cam, use_rgb=True)

    # 8. 화면에 띄우기
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Swin Transformer Attention (Grad-CAM)")
    plt.imshow(visualization)
    plt.axis('off')

    plt.show()

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
        test_result = metrics.compute()
        avg_val_loss = total_val_loss / len(val_DL)
        tqdm.write(f"Accurracy:{test_result['acc'].item():.3f} F1_score:{test_result['f1_score'].item():.3f}")
        val_history.append(avg_val_loss)

    tqdm.write(f"avg_train_loss:{avg_train_loss:.3f} avg_val_loss:{avg_val_loss:.3f}")


plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCH + 1), train_history, label='Train Loss', color='green', marker='o')
plt.plot(range(1, EPOCH + 1), val_history, label='Val Loss', color='red', marker='s')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(1, EPOCH + 1))
plt.legend()
plt.grid(True)
plt.show()
