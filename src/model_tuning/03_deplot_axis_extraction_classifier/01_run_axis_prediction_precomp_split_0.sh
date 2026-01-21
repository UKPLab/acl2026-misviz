#!/bin/bash
#
#SBATCH --job-name=misleader-dataset-precompute-all-embeddings
#SBATCH --mail-user=your_mail
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

source /miniconda3/etc/profile.d/conda.sh
conda activate lying_charts


srun python "src/model_tuning/03_deplot_axis_extraction_classifier/01_precompute_axis_extractions.py" \
    --model_path "data/deplot_finetune/output/deplot_finetune_accelerate/weights/lora_axis_adapter_3" \
    --datasetpath_misviz_synth "data/misviz_synth/" \
    --datasetpath_misviz "data/misviz/" \
    --outputpath "src/model_tuning/03_deplot_axis_extraction_classifier/output/raw_axis_deplot_axis_extraction" \
    --split 0