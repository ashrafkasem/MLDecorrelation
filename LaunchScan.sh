#!/bin/bash
#SBATCH --job-name=AdvScan    ### Job Name
#SBATCH --partition=gpu       ### Quality of Service (like a queue in PBS)
#SBATCH --time=0-04:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=7   ### Nuber of tasks to be launched per Node
#SBATCH --gres=gpu:1          ### General REServation of gpu:number of gpus
#SBATCH --mem=32G            ### Memory
#SBATCH --output=SlurmOuts/output_LA5D_AllLSR.txt
#SBATCH --error=SlrumErrors/error_LA5D_AllLSR.txt
#SBATCH --mail-user=bostdiek@uoregon.edu ### email for alerts
#SBATCH --mail-type=ALL

module load cuda/9.0
module load python3
