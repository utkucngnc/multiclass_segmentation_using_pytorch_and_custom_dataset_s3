import math
import torch
import json
import torchvision.transforms as transforms
from PIL import Image
from utils import labels as label_dict
from utils import dim, show_img_masks


class dataset_per_image():
    def __init__(self, root_im:str, root_ann:str, label_dict=label_dict, dim=dim):
        self.root_im = root_im
        self.root_ann = root_ann
        self.label_dict = label_dict
        self.dim = dim
        
        #Image data read & converted into Grayscale
        self.original_img = Image.open(self.root_im)
        self.original_img = self.original_img.convert("L")
        width, height = self.original_img.size
        
        self.img = self.original_img.resize((dim[2], dim[1]))
        
        self.img_tensor = transforms.ToTensor()(self.img)

        #Annotation file is read & loaded as a dictionary of dictionaries
        with open(root_ann, 'r') as f:
            self.annotation = json.load(f)
        
        #Normalization vector is created
        self.norm_factor = [1.0, 1.0]
        if self.dim:
            self.norm_factor[0] = float(self.dim[2] / width)
            self.norm_factor[1] = float(self.dim[1] / height)
        self.norm_factor = tuple(self.norm_factor)
        self.tensor = torch.zeros((len(self.label_dict),)+self.dim[1:])
        
    def __getitem__(self, pixel: tuple):
        return self.img[pixel]  #Return pixel value of the image
    
    def show(self, no_mask : bool):
        #Preview the image
        show_img_masks(self.img_tensor,self.tensor, no_mask)
    
    def __len__(self):
        #Returns shape of the image
        return self.img.shape[0]
    
    
    def find_bboxes(self):
        cats = self.annotation['categories']
        annos = self.annotation['annotations']
        bbox = {}
        for i in annos:
            ind = i['category_id']
            if not ind in bbox.keys():
                bbox[ind] = {}
                bbox[ind]['bbox'] = []
                bbox[ind]['name'] = ""
            [xmin, ymin, width, height] = i['bbox']
            xmax = (xmin+width)*self.norm_factor[0]
            ymax = (ymin+height)*self.norm_factor[1]
            xmin *=self.norm_factor[0]
            ymin *=self.norm_factor[1]
            
            xmax = math.floor(xmax)
            ymax = math.floor(ymax)
            ymin = math.floor(ymin)
            xmin = math.floor(xmin)
            
            norm_coord = [xmin,xmax,ymin,ymax]
            bbox[ind]['bbox'].append(norm_coord)
            
        for i in cats:
            bbox[i['id']]['name'] = i['name']
        return bbox            
        
    def build_mask(self, name):
        mask = torch.zeros(self.dim[1:], dtype=torch.float32)
        
        bbox = self.find_bboxes() # TODO: use the polygon here, not the bbox
        for elem in bbox:
            if bbox[elem]['name'] == name:
                for reg in bbox[elem]['bbox']:
                    slice_x = slice(reg[0],reg[1],1)
                    slice_y = slice(reg[2],reg[3],1)
                    mask[slice_y, slice_x] = 1
        return mask
        
    def build_tensor(self):
        for i in self.label_dict:
            label = self.label_dict[i]
            temp = self.build_mask(label)
            self.tensor[i] = temp
        return self.tensor