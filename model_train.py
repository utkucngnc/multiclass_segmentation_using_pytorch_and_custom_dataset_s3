import torch
from torch.utils.data import DataLoader

from utils import num_epoch
from dataset_sock import dataset_sock
from train import train_function

loss_vals = []
for num_batch in range(1, 26):
    epochs = range(1, num_epoch+1, 1)
    batch = dataset_sock(num_batch)

    train_dataloader = DataLoader(batch, batch_size=1, shuffle=True)

    for i, e in enumerate(epochs):
        print(f'Epoch: {e}')
        loss_val, unet, optimizer = train_function(train_dataloader)
        loss_vals.append(loss_val)
            
        torch.save({
            'model_state_dict': unet.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': e+1,
            'loss_values': loss_vals
        }, 'model.pth')
            
        print(f"Epoch {e} out of {num_epoch} completed. Model saved. Loss: {loss_val} ")
    print(f'Batch {num_batch} is done.')