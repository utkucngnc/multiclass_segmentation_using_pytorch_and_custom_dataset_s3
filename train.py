from UNet import UNet
from utils import device
import torch.nn as nn
import torch.optim as optim 
from tqdm import tqdm

model = UNet().to(device).train()
optimizer = optim.Adam(model.parameters(), lr = 0.0005)
loss_fn = nn.BCELoss()

def train_function(data, model=model, optimizer=optimizer, loss_fn=loss_fn, device=device):
    data = tqdm(data)
    
    for i, dat in enumerate(data):
        X,y = dat
        X,y = X.to(device), y.to(device)
        preds = model(X)
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss.item(), model, optimizer