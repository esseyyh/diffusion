import torch
import torchvision.transforms as transforms
import numpy as np
import torch
import matplotlib.pyplot as plt
import PIL
from PIL import Image

import os
import csv
from torch.utils.data import Dataset, DataLoader


# Create the custom dataset.

class ImageDataset(Dataset):
    def __init__(self, root_dir, csv_file):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform= transforms.Compose([
            #transforms.ToTensor(), # Convert to torch tensor (scales data into [0,1])
            transforms.Lambda(lambda t: (t/255)),
            transforms.Lambda(lambda t: (t * 2) - 1),
            transforms.Lambda(lambda t: t.permute([2,0,1])), # Scale data between [-1, 1] 
])



        with open(self.csv_file, "r") as f:
            reader = csv.reader(f)
            self.image_paths = []
            self.depth_paths = []
            for row in reader:
                image_,depth_ = row[0], row[1]
                self.image_paths.append(os.path.join(self.root_dir,image_))
                self.depth_paths.append(os.path.join(self.root_dir,depth_))
                
              

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.image_paths[index]
        depth= self.depth_paths[index]

        image = torch.from_numpy(np.array((Image.open(image))))
        depth = torch.from_numpy(np.array((Image.open(depth).convert("L")))).unsqueeze(2)
        image = self.transform(image)
        depth = self.transform(depth)

        return image, depth



     






#reverse_transform = transforms.Compose([
    #    transforms.Lambda(lambda t: (t + 1) / 2), # Scale data between [0,1]
    #    transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    #    transforms.Lambda(lambda t: t * 255.), # Scale data between [0.,255.]
    #    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)), # Convert into an uint8 numpy array
    #    transforms.ToPILImage(), # Convert to PIL image
#])
   
