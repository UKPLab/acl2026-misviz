#!/bin/bash
#
#SBATCH --job-name=misleader-dataset-encode-tables
#SBATCH --mail-user=your_mail
#SBATCH --mail-type=ALL
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_model:a100"

source /miniconda3/etc/profile.d/conda.sh
conda activate lying_charts


srun python "src/model_tuning/03_deplot_axis_extraction_classifier/02_encode_tables.py" \