import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import AutoEncoder
from load import CBIRDataset
import os


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
print("Base path:", BASE_PATH)
dataset = CBIRDataset(os.path.join(BASE_PATH, "DeepCBIR_Dataset"))
print("Dataset size:", len(dataset))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Entraînement
epochs = 20
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] completed, Average Loss: {avg_loss:.4f}")
    print("-" * 50)
    print(model.eval())

# Create model directory if it doesn't exist
os.makedirs(os.path.join(BASE_PATH, "model"), exist_ok=True)
torch.save(model.state_dict(), f"{BASE_PATH}/model/autoencoder.pth")