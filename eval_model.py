from dataset_sock import dataset_sock
from UNet import UNet
import matplotlib.pyplot as plt
import torch
from utils import model_path, device, show_img_masks
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


val_batch = dataset_sock(4, type='val')
val_loader = DataLoader(val_batch, batch_size = 1, shuffle=True)
checkpoints = torch.load(model_path)
model = UNet()
model.load_state_dict(checkpoints['model_state_dict'])
model.cuda()
model.eval()

@torch.inference_mode()
def evaluate():
    for i,elem in tqdm(enumerate(val_loader)):
        X, y = elem
        X, y = X.to(device), y.to(device)

        predictions = model(X)
        print(torch.argmin(predictions[0],dim=1))
        print(torch.argmax(predictions[0],dim=1))
        #show_img_masks(X[0],predictions[0],(2,4),(20,10))


evaluate()