import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,Subset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


import os
import hydra
from utils.data import ImageDataset,Sampling
from ldm.diffusion.unet import UNET
from scripts.trainer import Trainer

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)



def load_train_objs(cfg):
    model = AE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    return  model, optimizer


def prepare_dataloader(cfg):

    #
    dataset=ImageDataset(cfg.data.root_dir,cfg.data.csv_dir)
    train_indices = torch.arange(len(dataset))[:int(cfg.data.train_split * len(dataset))]
    test_indices = torch.arange(len(dataset))[int(cfg.data.train_split * len(dataset)):]
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_subset,pin_memory=True,shuffle=False, batch_size=1,sampler=DistributedSampler(train_subset))
    test_loader = DataLoader(test_subset, pin_memory=True, shuffle=False,batch_size=1,sampler=DistributedSampler(test_subset))

    return train_loader,test_loader



def main(rank: int, world_size: int,cfg):
    ddp_setup(rank, world_size)
    model, optimizer = load_train_objs(cfg)
    train_data,test_data = prepare_dataloader(cfg)
    trainer = Trainer(model, train_data,test_data, optimizer, rank, cfg.params.save_fre,cfg)
    trainer.train(1)
    destroy_process_group()



if __name__ == "__main__":   

    
    
    @hydra.main(version_base=None,config_path="config",config_name="config")
    def start(cfg):
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size,cfg), nprocs=world_size) 
    
    start()

