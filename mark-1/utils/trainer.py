import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,Subset
import torch  
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group



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
        loss = F.mse_los(output,data[:][2])
        loss.backward()
        self.optimizer.step()
        print("test run using diffusion data loader image encoding")
        print(f"loss : {loss}")
        print("---------")
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for data in self.train_data:
            self._run_batch(data)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(1):
            self._run_epoch(epoch)
            #if self.gpu_id == 0 and epoch % self.save_every == 0:
                #self._save_checkpoint(epoch)



