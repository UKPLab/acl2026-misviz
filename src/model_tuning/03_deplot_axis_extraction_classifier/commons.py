import os
import sys
import json
import torch

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_folder not in sys.path:
    sys.path.insert(0, root_folder)
import precomp_dataset


def prepare_datasets_misviz_synth(
    model_name,
    precomp_path,
    misviz_synth_path,
    output_prev_steps_path,
    dataset_type,
    label_to_idx,
    device,
):
    metadata_path = os.path.join(misviz_synth_path, "misviz_synth.json")
    with open(metadata_path, "r") as metadata_file:
        all_metadata = json.load(metadata_file)

    val_metadata = [entry for entry in all_metadata if "val" == entry["split"]]
    train_metadata = [entry for entry in all_metadata if "train_small" in entry["split"]]
    test_metadata = [entry for entry in all_metadata if "test" == entry["split"]]


    misviz_synth_precomp_path = precomp_path + "misviz_synth/"
    train_data = torch.load(
        misviz_synth_precomp_path
        + f"train_misviz_synth_{model_name}_embedded_images.pt",
        map_location=torch.device(device),
    )
    val_data = torch.load(
        misviz_synth_precomp_path + f"val_misviz_synth_{model_name}_embedded_images.pt",
        map_location=torch.device(device),
    )
    test_data = torch.load(
        misviz_synth_precomp_path
        + f"test_misviz_synth_{model_name}_embedded_images.pt",
        map_location=torch.device(device),
    )

    if dataset_type == "encoder_only":
        train_dataset = precomp_dataset.PrecompMisvizSynthDataset(
            train_metadata, train_data, label_to_idx
        )
        val_dataset = precomp_dataset.PrecompMisvizSynthDataset(
            val_metadata, val_data, label_to_idx
        )
        test_dataset = precomp_dataset.PrecompMisvizSynthDataset(
            test_metadata, test_data, label_to_idx
        )
    elif dataset_type == "with_axis":
        encoded_generated_tables_path = (
            output_prev_steps_path + "encoded_axis_extractions/"
        )

        misviz_synth_test_axis_encodings = torch.load(
            encoded_generated_tables_path
            + "misviz_synth_test_data_tapas_table_encodings_cls.pt",
            map_location=torch.device(device),
        )
        misviz_synth_val_axis_encodings = torch.load(
            encoded_generated_tables_path
            + "misviz_synth_val_data_tapas_table_encodings_cls.pt",
            map_location=torch.device(device),
        )
        misviz_synth_train_axis_encodings = torch.load(
            encoded_generated_tables_path
            + "misviz_synth_train_data_tapas_table_encodings_cls.pt",
            map_location=torch.device(device),
        )

        raw_axis_generations_path = (
            output_prev_steps_path + "raw_axis_deplot_axis_extraction/"
        )

        with open(
            raw_axis_generations_path
            + "misviz_synth_test_axis_deplot_axis_generations.json",
            "r",
        ) as test_raw_axis_data_file:
            test_raw_generations = json.load(test_raw_axis_data_file)

        with open(
            raw_axis_generations_path
            + "misviz_synth_train_small_axis_deplot_axis_generations.json",
            "r",
        ) as train_raw_axis_data_file:
            train_raw_generations = json.load(train_raw_axis_data_file)

        with open(
            raw_axis_generations_path
            + "misviz_synth_val_axis_deplot_axis_generations.json",
            "r",
        ) as val_raw_axis_data_file:
            val_raw_generations = json.load(val_raw_axis_data_file)

        train_dataset = precomp_dataset.PrecompMisvizSynthDatasetWithAxis(
            train_metadata,
            train_data,
            misviz_synth_train_axis_encodings,
            label_to_idx,
            train_raw_generations,
        )
        val_dataset = precomp_dataset.PrecompMisvizSynthDatasetWithAxis(
            val_metadata,
            val_data,
            misviz_synth_val_axis_encodings,
            label_to_idx,
            val_raw_generations,
        )
        test_dataset = precomp_dataset.PrecompMisvizSynthDatasetWithAxis(
            test_metadata,
            test_data,
            misviz_synth_test_axis_encodings,
            label_to_idx,
            test_raw_generations,
        )

    return train_dataset, val_dataset, test_dataset


def prepare_datasets_misviz(
    model_name,
    precomp_path,
    output_prev_steps_path,
    misviz_path,
    dataset_type,
    label_to_idx,
    device,
):
    metadata_path = os.path.join(misviz_path, "misviz.json")
    with open(metadata_path, "r") as metadata_file:
        all_metadata = json.load(metadata_file)

    test_metadata = [
        metadata_entry
        for metadata_entry in all_metadata
        if metadata_entry["split"] == "test"
    ]
    val_metadata = [
        metadata_entry
        for metadata_entry in all_metadata
        if metadata_entry["split"] == "val"
    ]

    misviz_precomp_path = precomp_path + "misviz/"

    val_data = torch.load(
        misviz_precomp_path + f"val_misviz_{model_name}_embedded_images.pt",
        map_location=torch.device(device),
    )
    test_data = torch.load(
        misviz_precomp_path + f"test_misviz_{model_name}_embedded_images.pt",
        map_location=torch.device(device),
    )

    if dataset_type == "encoder_only":

        val_dataset = precomp_dataset.PrecompMisvizDataset(
            val_metadata, val_data, label_to_idx
        )
        test_dataset = precomp_dataset.PrecompMisvizDataset(
            test_metadata, test_data, label_to_idx
        )
    elif dataset_type == "with_axis":
        encoded_generated_tables_path = (
            output_prev_steps_path + "encoded_axis_extractions/"
        )

        misviz_test_axis_encodings = torch.load(
            encoded_generated_tables_path
            + "misviz_test_data_tapas_table_encodings_cls.pt",
            map_location=torch.device(device),
        )
        misviz_val_axis_encodings = torch.load(
            encoded_generated_tables_path
            + "misviz_val_data_tapas_table_encodings_cls.pt",
            map_location=torch.device(device),
        )

        raw_axis_generations_path = (
            output_prev_steps_path + "raw_axis_deplot_axis_extraction/"
        )

        with open(
            raw_axis_generations_path + "misviz_test_axis_deplot_axis_generations.json",
            "r",
        ) as test_raw_axis_data_file:
            test_raw_generations = json.load(test_raw_axis_data_file)

        with open(
            raw_axis_generations_path + "misviz_val_axis_deplot_axis_generations.json",
            "r",
        ) as val_raw_axis_data_file:
            val_raw_generations = json.load(val_raw_axis_data_file)

        val_dataset = precomp_dataset.PrecompMisvizDatasetWithAxis(
            val_metadata,
            val_data,
            misviz_val_axis_encodings,
            label_to_idx,
            val_raw_generations,
        )
        test_dataset = precomp_dataset.PrecompMisvizDatasetWithAxis(
            test_metadata,
            test_data,
            misviz_test_axis_encodings,
            label_to_idx,
            test_raw_generations,
        )

    return val_dataset, test_dataset
