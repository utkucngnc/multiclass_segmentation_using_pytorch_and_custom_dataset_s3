import os
from minio import Minio
import shutil
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torch_dev import dev

labels = {
    0: 'Staine',
    1: 'MissingString',
    2: 'ReducedDefect',
    3: 'Hole',
    4: 'OpenEnd',
    5: 'Text',
    6: 'Needlemistake'
    }

dim = (1,640,800) # 1 H W
work_dir = os.getcwd()
train_path = f'{work_dir}/train'
val_path = f'{work_dir}/val'
test_path = f'{work_dir}/test'
model_path = f'{work_dir}/model.pth'
num_epoch = 10
device = dev()

bucket_name = "measuringobjects.p108"
access_key = 'minioadmin'
secret_key = 'minioadmin'
ip = '192.168.100.222:9000'

def connect():
    client = Minio(ip,
            access_key,
            secret_key,
            secure=False)
    return client
    

#Some auxillary functions

def is_png(obj_name):
    if type(obj_name) != str:
        obj_name = str(obj_name)
    
    if(obj_name[-4:]=='.png'):
        return True
    else:
        return False

def is_json(obj_name):
    if type(obj_name) != str:
        obj_name = str(obj_name)
    
    if(obj_name[-5:]=='.json'):
        return True
    else:
        return False

def show_img_masks(img_tensor:None, mask_tensors:None, grid:None,
                   fig_size:None, label_dict=labels,no_masks = False):

    if not no_masks:
        tensors = torch.cat((img_tensor, mask_tensors))
        if device == 'gpu':
            tensors = tensors.to('cpu')
        plt.figure(figsize=fig_size)
        
        for i in label_dict:
            label = label_dict[i]
            if i==0:
                plt.subplot(grid[0],grid[1],i+1)
                img = transforms.ToPILImage()(tensors[i])
                plt.imshow(img)
                plt.gca().set_title(f'Original Image')
            plt.subplot(grid[0],grid[1],i+2)
            plt.imshow(tensors[i+1])
            plt.gca().set_title(f'Mask for {label}')
        plt.show()
    else:
        img = transforms.ToPILImage()(img_tensor)
        plt.imshow(img)
        plt.title("Original Image")
        plt.show()

def check_folders(train_path = train_path, val_path = val_path, test_path = test_path,
                  remove = True):
    files = [train_path, val_path, test_path]
    
    for file in files:
        if os.path.isdir(file):
            if remove:
                shutil.rmtree(file)
                os.mkdir(file)
        else:
            try:
                os.mkdir(file)
            except FileExistsError:
                pass