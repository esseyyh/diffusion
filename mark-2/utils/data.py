import torch
import os
import csv
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from transformers import CLIPTokenizer
from scripts.clip_model_loader import load_models_
from scripts.ae_model_loader import load_decoder
from scripts.ae_model_loader import load_encoder
from datetime import datetime


from ldm.dpm.ddpm import DDPM
from ldm.vae.encoder import Encoder
from ldm.vae.decoder import Decoder
from ldm.clip.clip import CLIP



# Create the custom dataset.
class ImageDataset(Dataset):
    def __init__(self, root_dir, csv_file):

        # load the folder containing the dataset and the csv file with the captions
        self.root_dir = root_dir
        self.csv_file = csv_file

        # random number generator for generating random number for diffusion 
        gen=torch.Generator()
        self.diff=DDPM(gen)

        # tokenizing the caption 
        self.tokenizer = CLIPTokenizer("utils/vocab.json", merges_file="utils/merges.txt")

        # loading the models  and thier parameters 

        
        self.txt_encoder=load_models_()
        self.time_steps=int(self.diff.num_train_timesteps)
        self.encoder=load_encoder()

        # simpel preproccesing

        self.transform= transforms.Compose([
            transforms.Lambda(lambda t: t.permute(2,0,1)),
            transforms.Lambda(lambda t: (t/255)),
            transforms.Lambda(lambda t: ((t*2)-1)), # Scale data between [-1, 1] 
            ])


        # inputting the data captions names and path in lists 
        with open(self.csv_file, "r") as f:
            reader = csv.reader(f)
            self.image_paths = []
            self.txt=[]
            
            for row in reader:
                image,txt = row[1], row[3]
                self.image_paths.append(os.path.join(self.root_dir,image))
                self.txt.append(txt)
                
                
              

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        # image or batch index as input 
        image = self.image_paths[index]


        # random time step generation 
        t = torch.randint(0, self.time_steps,(1,))
        time_embedding=self.diff.time_embedding(int(t))
        

        # turn image to tensor

        image =plt.imread(image)
        image=(self.transform(torch.tensor(image,dtype=torch.float32))).unsqueeze(0)

        # test and image encoding  
        

        with torch.no_grad():
        
            noisy_image=self.diff.add_noise(image,t)
            noisy_image=self.encoder(noisy_image)
            image=self.encoder(image)
            text_guiance=self.txt_encoder(torch.tensor(self.tokenizer.batch_encode_plus([self.txt[index]],padding="max_length", max_length=77).input_ids))
            

        
   

        return image,noisy_image,time_embedding,text_guiance


class Sampling():
    def __init__(self,save_dir=None):

        self.dir=save_dir
        self.decoder=load_decoder()
        

        self.transform= transforms.Compose([

            transforms.Lambda(lambda t: t.permute([1,2,0])),
            transforms.Lambda(lambda t: (t + 1)/2 ),
            transforms.Lambda(lambda t: (t*255)),           # reverse the scaling process 
            ])
        
    def __call__(self,x,filename):
        with torch.no_grad():

            x=self.decoder(x)
            

        if self.dir != None:
            
            if (x.shape)[0] >1:
                
                for i in range ((x.shape)[0]):
                    print("running te latest code")
                    

                    image=x[i].clamp(-1,1)
                    image=torch.tensor(self.transform(image),dtype=torch.uint8)
                    image=image.numpy()
                    name=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    plt.imsave(f"samples/{filename}+{i}+{name}.jpg",image)

                    # image = Image.fromarray(image)
                    
                    # image.save(f"samples/{name}.jpg")
            else:
                   
                    x=x.squeeze(0)
                    x=x.clamp(-1,1)
                    image=self.transform(x)
                    image=torch.tensor(image,dtype=torch.uint8)
                    image=image.numpy()
                    name=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    plt.imsave(f"samples/{name}.jpg",image)

                    # image=(self.transform(x)).numpy()
                    
                    # image = Image.fromarray(image)
                    # image.save(f"samples/{name}")
        else:
            x=x.squeeze(0)
            x=x.clamp(-1,1)
            image=self.transform(x)
           
            image=torch.tensor(image,dtype=torch.uint8)
            

            # image = Image.fromarray(image)
            return image.numpy()

            







