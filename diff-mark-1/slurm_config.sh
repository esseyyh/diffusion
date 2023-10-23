#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=train_check
#SBATCH --time=22:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=teach0013

module load miniconda/3 -q
module load cuda/11.4 -q

conda create --name check python==3.8 -y -q

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
pip3 install hydra-core -q
pip3 install numpy -q 
pip3 install matplotlib -q
pip3 install pillow -q
python train.py  hydra/job_logging=disabled  1> out/out.txt 2>out/err.txt
