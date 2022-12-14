#!/bin/bash -x

#SBATCH --output=encas-nat-refss-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=47:59:00
#SBATCH --mem=192GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=encas-nat-refss
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME
echo $CUDA_VISIBLE_DEVICES

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash /scratch/bf996/encas/script/run-singularity-encas-cuda.bash \
    /bin/bash -c \
    'python nat_run_many.py --config configs_nat/cifar100_reproducenat.yml'