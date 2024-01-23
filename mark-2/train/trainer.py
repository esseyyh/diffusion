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
import numpy as np
from utils.data import ImageDataset
from  src.autoencoder.ae  import AE 


class Trainer:
    def __init__(self,model: torch.nn.Module,test_data: DataLoader,train_data: DataLoader,optimizer: torch.optim.Optimizer,gpu_id: int,save_every: int,cfg ) -> None:
        self.gpu_id = gpu_id 
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.test_data=test_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])


    def _run_batch(self, data,epoch,gpu_id):
        self.optimizer.zero_grad()

        data1 = data[:].to(self.gpu_id)
        output = self.model(data1)
        loss = F.mse_loss(output,data1)
        loss.backward()
        self.mean_train_loss.append(loss.item)
        self.optimizer.step()

        self.optimizer.zero_grad()
        print(f"{epoch} ,{gpu_id},{loss }" )
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        #print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        self.mean_train_loss=[]
        for data in self.train_data:
            
            self._run_batch(data,epoch,self.gpu_id)
        #print(f"loss : {np.mean(self.mean_train_loss)}")
    def _run_test_batch(self, data):
        self.optimizer.zero_grad()

        data1 = data[:][0].to(self.gpu_id)
        output = self.model(data1,True,False)
        loss = F.mse_loss(output,data1)
        loss.backward()
        self.mean_test_loss.append(loss.item)
        self.optimizer.zero_grad()
        data2 = data[:][1].to(self.gpu_id)
        output = self.model(data2,True,False)
        loss = F.mse_loss(output,data2)
        loss.backward()
        self.mean_test_loss.append(loss.item)
        #print(f"Loss : {loss }" )

    def _run_test_epoch(self, epoch):
        b_sz = len(next(iter(self.test_data))[0])
        print(f"Testing [GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.test_data)}")
        self.test_data.sampler.set_epoch(epoch)
        self.mean_test_loss=[]
        for data in self.test_data:
            
            self._run_test_batch(data)

        print(f"test loss : np.mean(self.mean_test_loss)") 
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "/home/kunet.ae/100053688/out/hpc_tasks_ae/trail_1/checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            #if self.gpu_id == 0 and epoch % self.save_every == 0:
                #self._run_test_epoch(1)
                #self._save_checkpoint(epoch)
