import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,Subset
import torch  
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import hydra
from utils.data import ImageDataset
from  src.auto import AE 

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

class Trainer:
    def __init__(self,model: torch.nn.Module,train_data: DataLoader,optimizer: torch.optim.Optimizer,gpu_id: int,save_every: int,cfg ) -> None:
        self.gpu_id = gpu_id 
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, data):
        self.optimizer.zero_grad()

        data1 = data[:][1].to(self.gpu_id)
        output = self.model(data1)
        loss = F.mse_loss(output,data[:][2])
        loss.backward()
        self.optimizer.step()


    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for data in self.train_data:
            
            self._run_batch(data)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "/home/kunet.ae/100053688/out/hpc_tasks_diff/trail_1/checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs(cfg):
    model = AE(cfg.model_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.LR_1)
    return  model, optimizer


def prepare_dataloader(cfg):

    #
    dataset=ImageDataset(cfg.data.root_dir,cfg.data.csv_dir)
    train_indices = torch.arange(len(dataset))[:int(cfg.data.train_split * len(dataset))]
    test_indices = torch.arange(len(dataset))[int(cfg.data.train_split * len(dataset)):]
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_subset,pin_memory=True,shuffle=False, batch_size=cfg.data.batch_size,sampler=DistributedSampler(train_subset))
    test_loader = DataLoader(test_subset, pin_memory=True, shuffle=False,batch_size=cfg.data.batch_size,sampler=DistributedSampler(test_subset))

    return train_loader,test_loader



def main(rank: int, world_size: int,cfg):
    ddp_setup(rank, world_size)
    model, optimizer = load_train_objs(cfg)
    train_data,test_data = prepare_dataloader(cfg)
    trainer = Trainer(model, train_data, optimizer, rank, cfg.params.save_fre,cfg)
    trainer.train(cfg.params.no_epoch)
    destroy_process_group()



if __name__ == "__main__":   

    
    
    @hydra.main(version_base=None,config_path="config",config_name="config")
    def start(cfg):
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size,cfg), nprocs=world_size) 
    
    start()

