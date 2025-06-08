import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image, ImageFilter
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Функция для отображения изображений с результатами
def imshow_pairs(imgs, labels, distances):
    """Отображает пары изображений на одной строке с текстовыми метками."""
    fig, axs = plt.subplots(1, len(imgs), figsize=(10 * len(imgs), 5))
    if len(imgs) == 1:
        axs = [axs]

    for idx, (img, label, dist) in enumerate(zip(imgs, labels, distances)):
        axs[idx].imshow(np.transpose(img.numpy(), (1, 2, 0)), cmap="gray")
        axs[idx].axis("off")
        axs[idx].set_title(f"{label}: {dist:.2f}", fontsize=10, color="green" if "Similarity" in label else "red")

    plt.tight_layout()
    plt.show()

# Определение сиамской сети
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 37 * 37, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_one(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        return output1, output2

# Функция контрастной утраты
def contrastive_loss(output1, output2, label, margin=1.0):
    euclidean_distance = F.pairwise_distance(output1, output2)
    loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                      label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss

# Путь к датасету
train_data_path = "/content/drive/MyDrive/architecture_kazan"
folder_dataset_train = datasets.ImageFolder(root=train_data_path)
class_names = folder_dataset_train.classes
class_to_idx = folder_dataset_train.class_to_idx

# Преобразование для обучения
transformation = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

# Генерация пар для выборки
def create_sample_pairs(dataset, num_pairs=125):
    """Создаёт выборку пар изображений с метками."""
    pairs = []
    class_names = dataset.classes
    class_to_idx = dataset.class_to_idx

    for _ in range(num_pairs // 2):
        # Пара из одного стиля
        while True:
            arch_class = random.choice(class_names)
            arch_images = [img for img, label in dataset.imgs if label == class_to_idx[arch_class]]
            if len(arch_images) > 1:
                img_path1 = random.choice(arch_images)
                img_path2 = random.choice(arch_images)
                if img_path1 != img_path2:
                    img1 = Image.open(img_path1).convert("RGB")
                    img2 = Image.open(img_path2).convert("RGB")
                    pairs.append((img1, img2, 0.0))
                    break

        # Пара из разных стилей
        while True:
            arch_class1 = random.choice(class_names)
            arch_class2 = random.choice(class_names)
            if arch_class1 != arch_class2:
                arch_images1 = [img for img, label in dataset.imgs if label == class_to_idx[arch_class1]]
                arch_images2 = [img for img, label in dataset.imgs if label == class_to_idx[arch_class2]]
                img_path1 = random.choice(arch_images1)
                img_path2 = random.choice(arch_images2)
                img1 = Image.open(img_path1).convert("RGB")
                img2 = Image.open(img_path2).convert("RGB")
                pairs.append((img1, img2, 1.0))
                break

    return pairs

# Создание модели
net = SiameseNetwork()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Генерация пар для выборки
pairs = create_sample_pairs(folder_dataset_train, num_pairs=125)

# Обучение модели
print("Начало обучения модели")
for epoch in range(5):
    total_loss = 0
    for pair in pairs:
        img1, img2, label = pair
        img1_t = transformation(img1).unsqueeze(0)
        img2_t = transformation(img2).unsqueeze(0)
        label_t = torch.tensor([label], dtype=torch.float32)

        optimizer.zero_grad()
        output1, output2 = net(img1_t, img2_t)
        loss = contrastive_loss(output1, output2, label_t)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/5], Loss: {total_loss / len(pairs):.4f}")

# Тестирование на изображениях с фильтром Собеля
print("Тестирование на изображениях с фильтром Собеля:")
sobel_transformation = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.Grayscale(num_output_channels=3), # Преобразование изображения в оттенки серого
    transforms.ToTensor()
])

for idx, pair in enumerate(pairs[:5]):
    img1, img2, label = pair

    # Применение фильтра Собеля
    img1_sobel = Image.fromarray(np.array(img1.filter(ImageFilter.FIND_EDGES)))   #Для каждого изображения из пары применяется фильтр Собеля (через ImageFilter.FIND_EDGES), который выделяет края на изображении.
    img2_sobel = Image.fromarray(np.array(img2.filter(ImageFilter.FIND_EDGES)))

    #К изображениям с фильтром Собеля применяются ранее созданные преобразования (изменение размера, преобразование в оттенки серого и ToTensor)
    img1_t = sobel_transformation(img1_sobel).unsqueeze(0)
    img2_t = sobel_transformation(img2_sobel).unsqueeze(0)
    output1, output2 = net(img1_t, img2_t) #Получение выходов модели и вычисление расстояния
    distance = F.pairwise_distance(output1, output2).item()

    # Печать результата
    print(f"Пара {idx + 1}: Label = {label}, Distance = {distance:.2f}")

    # Отображение изображений
    concatenated = torch.cat((sobel_transformation(img1_sobel), sobel_transformation(img2_sobel)), dim=2)
    plt.imshow(np.transpose(concatenated.numpy(), (1, 2, 0)), cmap="gray")
    plt.title(f"Label: {label}, Distance: {distance:.2f}")
    plt.axis("off")
    plt.show()
