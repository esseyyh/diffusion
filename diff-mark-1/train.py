import torch
import numpy as np
import hydra
from torch.utils.data import DataLoader,Dataset,Subset

from src.network.unet import UNet
from utils.data import ImageDataset



@hydra.main(version_base=None,config_path="config",config_name="config")
def train (cfg):
    dataset=ImageDataset(cfg.data.root_dir,cfg.data.csv_dir)
    
    
    train_indices = torch.arange(len(dataset))[:int(cfg.data.train_split * len(dataset))]
    test_indices = torch.arange(len(dataset))[int(cfg.data.train_split * len(dataset)):]


    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

# Create data loaders for the training and test subsets
    train_loader = DataLoader(train_subset,shuffle= cfg.data.train_shuffle, batch_size=cfg.data.batch_size)#,num_workers=4)
    test_loader = DataLoader(test_subset, shuffle=cfg.data.test_shuffle, batch_size=cfg.data.batch_size)#,num_workers=4)

    model = UNet.to("cuda:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.LR_1)
    for epoch in range(cfg.params.no_epoch):
        mean_epoch_loss=[]
        for batch in train_loader:



            
    
            data = batch[:][1]
            data=data.to("cuda:0") 
            images = model(data,train,False)
            outs = batch[:][2]
            outs=outs.to("cuda:0")
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(images,outs) 
            mean_epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
    
        if epoch % cfg.params.save_fre == 0:
            print('---')
            print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss)}")
            torch.save(model,"model.pt")

    print("#######")
    print("#################")
    print("#######")
    #for epoch in range(cfg.params.no_epoch):
    #    mean_epoch_loss=[]
    #    for batch in test_loader:



            
    
     #       batch_image  = batch
     #       batch_image=batch_image.to("cuda:0")
  

     #       images = model(batch_image,train,False)
    
     #       optimizer.zero_grad()
     #       loss = torch.nn.functional.mse_loss(batch_image,images) 
     #       mean_epoch_loss.append(loss.item())
  
     #   if epoch % cfg.params.no_epoch ==0:
     #           print('---')
     #           print (f"Epoch :{epoch}|testloss {np.mean(mean_epoch_loss)} ")
    


if __name__=="__main__":
    train()

































#image2=Image.open("image2.jpg")
#image1=Image.open("image1.jpg")


#transform = transforms.Compose([
    #transforms.ToTensor(), # Convert to torch tensor (scales data into [0,1])
    #transforms.Lambda(lambda t: (t * 2) - 1), # Scale data between [-1, 1] 
#])


#reverse_transform = transforms.Compose([
    #transforms.Lambda(lambda t: (t + 1) / 2), # Scale data between [0,1]
    #transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
   # transforms.Lambda(lambda t: t * 255.), # Scale data between [0.,255.]
  #  transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)), # Convert into an uint8 numpy array
 #   transforms.ToPILImage(), # Convert to PIL image
#])

#torch_image2 = transform(image2)
#torch_image1 = transform(image1)






