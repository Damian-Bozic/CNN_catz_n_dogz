import torch
import torch.nn as nn
import torch.nn.functional as F


class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()

        # Wejście: 3 kanały RGB, 64x64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Nowa, głębsza warstwa wyciągająca detale
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # Po 4 warstwach MaxPool: 64 -> 32 -> 16 -> 8 -> 4x4 piksele
        # Nowy rozmiar wektora: 128 kanałów * 4 * 4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 4x4

        x = x.view(-1, 128 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = CatDogCNN()
    random_input = torch.randn(1, 3, 64, 64)
    output = model(random_input)
    print(f"Nowy model działa! Kształt wyjścia: {output.shape}")
