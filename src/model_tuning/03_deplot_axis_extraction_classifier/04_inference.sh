#!/bin/bash
#
#SBATCH --job-name=misleader-dataset-inference
#SBATCH --mail-user=your_mail
#SBATCH --mail-type=ALL
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_model:a100"

source /miniconda3/etc/profile.d/conda.sh
conda activate lying_charts


srun python "viz_FC/src/model_tuning/03_deplot_axis_extraction_classifier/04_inference_classifier.py" \