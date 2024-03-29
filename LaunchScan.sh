#!/bin/bash
#SBATCH --job-name=Adv2p    ### Job Name
#SBATCH --partition=gpu       ### Quality of Service (like a queue in PBS)
#SBATCH --time=0-18:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=7   ### Nuber of tasks to be launched per Node
#SBATCH --gres=gpu:1          ### General REServation of gpu:number of gpus
#SBATCH --mem=32G             ### Memory
#SBATCH --array=0-9           ### Arrays for lambda
#SBATCH --output=SlurmOuts/output_2p_%a.txt
#SBATCH --error=SlrumErrors/error_2p_%a.txt
#SBATCH --mail-user=bostdiek@uoregon.edu ### email for alerts
#SBATCH --mail-type=ALL

PRONG=2

module load cuda/9.0
module load python3

echo $SLURM_NTASKS

export OMP_NUM_THREADS=$SLURM_NTASKS

cd /projects/het/bostdiek/Decorellation/

python3 ScanAdversaries.py $PRONG ${SLURM_ARRAY_TASK_ID}
