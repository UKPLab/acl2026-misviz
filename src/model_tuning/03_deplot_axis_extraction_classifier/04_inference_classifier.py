import argparse
import os
import sys
import json
import torch
from tqdm.auto import tqdm
import commons

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_folder not in sys.path:
    sys.path.insert(0, root_folder)

import utils

def load_model(base_model, e_type, seed, input_dim, hidden_dim, output_dim, device):
    base_path =f"src/model_tuning/03_deplot_axis_extraction_classifier/output/classifier_training/{base_model}_{e_type}/{base_model}_{e_type}_{seed}/weights"
    checkpoint_path = os.path.join(base_path, f"best_model.pth")
    model = utils.ClassifierHead(input_dim, hidden_dim, output_dim)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_inference(
    model,
    dataset,
    idx_to_label,
    device,
):
    results = []

    with torch.no_grad():
        for embedding, _, metadata in tqdm(dataset):
            outputs = model(embedding.unsqueeze(0).to(device))
            pred_idx = torch.argmax(outputs, dim=1).item()

            results.append(
                {
                    "image_path": metadata["image_path"],
                    "true_misleader": metadata.get("misleader", []),
                    "predicted_misleader": idx_to_label[pred_idx],
                }
            )

    return results


def main(args):
    device = utils.get_available_device()
    output_root = "src/model_tuning/results"

    label_to_idx = {'no misleader': 0,
        'discretized continuous variable': 1,
        'dual axis': 2,
        'inappropriate axis range': 3, 
        'inappropriate item order':  4,
        'inappropriate use of line chart': 5,
        'inappropriate use of pie chart': 6, 
        'inconsistent binning size' : 7,
        'inconsistent tick intervals': 8,
        'inverted axis': 9,
        'misrepresentation': 10, 
        'truncated axis': 11,
        '3d': 12
    }
    idx_to_label = {v:k for k,v in label_to_idx.items()}
    base_model= "tinychart"
    hidden_dim = 1024
    for seed in [123, 456, 789]:
        for e_type in ["encoder_only", "with_axis"]:
            _, _, test_dataset_misviz_synth = commons.prepare_datasets_misviz_synth(
                base_model,
                args.precomp_path,
                args.datasetpath_misviz_synth,
                args.output_prev_steps_path,
                e_type,
                label_to_idx,
                device,
            )

            _, test_dataset_misviz = commons.prepare_datasets_misviz(
                base_model,
                args.precomp_path,
                args.datasetpath_misviz,
                args.output_prev_steps_path,
                e_type,
                label_to_idx,
                device,
            )

            # Load model
            input_dim = test_dataset_misviz_synth.input_length()
            output_dim = len(label_to_idx)

            model = load_model(base_model, e_type, seed,
                input_dim,
                hidden_dim,
                output_dim,
                device,
            )

            # Inference
            results_synth = run_inference(
                model,
                test_dataset_misviz_synth,
                idx_to_label,
                device,
            )

            results_misviz = run_inference(
                model,
                test_dataset_misviz,
                idx_to_label,
                device,
            )

            os.makedirs(output_root, exist_ok=True)
            os.makedirs(os.path.join(output_root, f"{base_model}_{e_type}_{seed}"), exist_ok=True)

            with open(os.path.join(output_root, f"{base_model}_{e_type}_{seed}" ,"misviz_synth.json"), "w") as f:
                json.dump(results_synth, f, indent=2)

            with open(os.path.join(output_root, f"{base_model}_{e_type}_{seed}" , "misviz.json"), "w") as f:
                json.dump(results_misviz, f, indent=2)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetpath_misviz_synth", type=str, default="data/misviz_synth/")
    parser.add_argument("--datasetpath_misviz", type=str, default="data/misviz/")
    parser.add_argument("--precomp_path", type=str,default="data/precomp/",help="Path at which the precomputed misleader dataset is located at.",)
    parser.add_argument("--output_prev_steps_path", type=str, default="src/model_tuning/03_deplot_axis_extraction_classifier/output/", help="Path at which the output of the previous two steps is located at")

    args = parser.parse_args()
    main(args)
