#!/bin/bash
#SBATCH -J unet_test
#SBATCH -o %x.%j.out
##SBATCH -e %x.j%.err
#SBATCH -p toreador
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
##SBATCH --mem-per-cpu=1190MB
#SBATCH -t 01:00:00
##SBATCH --gpus-per-node=3 ##(for matador GPU only)
#SBATCH --gres=gpu:3
#SBATCH --mail-user=wil16363@ttu.edu
#SBATCH --mail-type=ALL
#module load toreador/0.15.4
#module load gcc/9.3.0 openmpi/3.1.6-cuda
#module load gcc/10.2.0

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiYmU1ODI4ZC1hZTg0LTQ1NWMtOTM4OS1iYzdkZjllYjBjODkifQ=="

#python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

#module load gcc/10.1.0 cuda/11.3.0 cudnn/8.2.0.53-1
module avail  

nvidia-smi

# Activate Conda environment correctly
export CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate test
conda info

cd ML_Eyes/
nvidia-smi

#nvcc hello.cu -o hello

#export CUDA_VISIBLE_DEVICES=0,1,2 
#srun python test.py
#srun ./hello
#srun python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

#python test.py
singularity exec --nv tensorflowgpu.sif python3 train.py --multi_gpu --seed 500 --model_type unet_2d --n_trials 30 --epochs 5 --batch_size 16 --optimizer adam --loss dice --im_width 256 --im_height 256  --learning_rate 0.0001 0.02 --n_filters 32 96  --layer_depth 4 6 --stack_num_down 1 2 --stack_num_up 1 2 --activation GELU --output_activation Sigmoid --use_neptune --use_optuna
