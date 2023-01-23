import torch
import os
import shutil
from torch.utils.data import Dataset
from utils import labels, dim, work_dir, bucket_name, train_path, val_path, test_path
from utils import check_folders, is_json, is_png, connect
from dataset_per_image import dataset_per_image as im_dat


class dataset_sock(Dataset):
    def __init__(
        self, batch_num : int, bucket_name=bucket_name, root=work_dir, 
        label_dict=labels, dim=dim, type = 'train', download_enabled = True, 
        ):
        self.client = connect()
        self.bucket_name = bucket_name
        self.root = root
        self.batch_num = batch_num
        self.dim = dim
        self.label_dict = label_dict
        self.len = 0
        self.type = type
        self.path = self.root
        self.im_paths = []
        self.eval = eval
        
        if self.type == 'train':
            self.path = train_path
        elif self.type == 'val':
            self.path = val_path
        elif self.type == 'test':
            self.path = test_path
        else:
            print('Check your dataset type.')
        
        check_folders(remove=download_enabled)
        
        if download_enabled:
            self.len = self.batch_downloader()
        else:
            self.len = self.batch_controller()
        
        self.gnd_truth= torch.zeros((self.len,len(self.label_dict),)+dim[1:])
        self.imgs = torch.zeros((self.len,)+dim)
        
        self.batch_loader()
    
    def __getitem__(self,index):
        #Returns the image and the ground truth corresponding to the index given
        return self.imgs[index], self.gnd_truth[index]

    def __len__(self):
        #Returns number of images in the batch
        return self.len    

    def batch_downloader(self):
        client = self.client
        cwd_temp = self.path
        print(cwd_temp)
        #Image and Annotation path strings
        im_path = ""
        ann_path = ""

        #Number of images and annotations downloaded
        num_ann = 0
        num_im = 0
        
        #Get object handle from MinIO Client
        objects = client.list_objects(self.bucket_name, prefix=f'new_data/{self.batch_num}Batch/', recursive=True)
        
        for obj in objects:
            if (os.getcwd()!=cwd_temp):
            #Use / Create a temporary directory
                os.chdir(cwd_temp)
            
            #Object name returns a string containing the path in the bucket
            obj_name = obj.object_name
            
            if (is_png(obj_name) or is_json(obj_name)):
                #Store the old path on local directory
                old_destination = f'{cwd_temp}/{obj_name}'
                
                #Download the file
                client.fget_object(self.bucket_name,obj_name, obj_name)
                    
                #Change the name & move it out of subdirectories for convenience
                #Remove the residual empty directory
                if is_png(obj_name):
                    im_path = f'{cwd_temp}/{num_im}_im.png'
                    self.im_paths.append(im_path)
                    shutil.move(old_destination,im_path)
                    shutil.rmtree(f'{cwd_temp}/new_data')
                    num_im+=1
                elif is_json(obj_name):
                    ann_path = f'{cwd_temp}/{num_ann}_anno.json'
                    shutil.move(old_destination,ann_path)
                    shutil.rmtree(f'{cwd_temp}/new_data')
                    num_ann+=1
            
            #Go to the main directory where the script runs
            os.chdir(self.root)
            
            if (im_path!="" and ann_path!=""):
                im_path = ""
                ann_path = ""
                print(f'{im_path}\n{ann_path}')
        
        if num_im == num_ann:
            print(f'{num_im} images and annotations are downloaded')
        else:
            print(f'Missing annotations / images are found: {abs(num_im - num_ann)}')
        return num_im
    
    def batch_controller(self):
        #used if the batch is already downloaded
        #Check the batch
        cwd_temp = self.path
        files = os.listdir(cwd_temp)
        
        im_num = 0
        ann_num = 0
        
        #Count images and annotations
        for file in files:
            if file[-1]=='g':
                im_num += 1
            elif file[-1] == 'n':
                ann_num += 1
        
        if im_num == ann_num and im_num!=0:
            print('Files were checked. Proceeding....')
        else:
            print('Missing files are found. Initializing the download sequence....')
            os.chdir(self.root)
            shutil.rmtree(cwd_temp)
            im_num = self.batch_downloader()
        return im_num
                 
        
    def batch_loader(self):
        #Loads the images and their annotations into output tensor
        cwd_temp = self.path
        
        for i in range(self.len):
            root_im = f'{cwd_temp}/{i}_im.png'
            root_ann = f'{cwd_temp}/{i}_anno.json'
            temp = im_dat(root_im=root_im, root_ann=root_ann)
                
            self.gnd_truth[i] = temp.build_tensor()
            self.imgs[i] = temp.img_tensor