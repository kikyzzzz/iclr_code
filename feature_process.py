import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import pickle

DATA_CONTAIN_BASE = "data_contain_base_directory"


def load_single_json(json_path, epoch, train_mode="lora"):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        index = data['metadata']['sample_info']['sample_index']
        label = data['metadata']['sample_info']['sample_label']

        epoch_data = data['gradient_update_history'][epoch]
        layers = epoch_data['layer_updates']
        output_update = None
        if "null" in layers:
            if 'embedding' not in layers["null"]:
                output_update, layernorm_update, embedding_update = layers["null"]['output']['avg_update'], layers["null"]['layernorm']['avg_update'], 0
            else:
                output_update, layernorm_update, embedding_update = layers["null"]['output']['avg_update_abs'], layers["null"]['layernorm']['avg_update_abs'], layers["null"]['embedding']['avg_update_abs']
            del layers["null"]

        features = []
        target_feature = 'avg_update'
        target_feature_abs = 'avg_update_abs'
        for layer_idx in sorted(layers.keys(), key=lambda x: int(x)):
            layer = layers[layer_idx]

        features_np = np.array(features)

        features_np[np.isinf(features_np)] = 0.01
        features_np[np.isnan(features_np)] = 0

        return features_np, label, index

    except Exception as e:
        return None, None, None


def load_single_json_upgrade(
        json_path,
        epoch,
        train_mode="lora",
        model_name="pythia-6.9b",
        selected_layers=None,
        selected_modules=None,
        selected_features=None,
        feat_trans=True
):
    if model_name == "pythia-6.9b":
        selected_modules = ["query_key_value", "dense_h_to_4h", "dense_4h_to_h", "dense"]
        selected_features = ['grad_abs_mean', 'grad_std',
                             'grad_top10p_ratio', 'grad_row_mean_std', 'grad_row_mean_max',
                             'grad_sparsity', 'grad_top10_row_eccentricity', 'grad_top10_col_eccentricity']
    elif model_name == "opt-6.7b":
        selected_modules = ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"]
        selected_features = ['grad_abs_mean', 'grad_std',
                             'grad_top10p_ratio', 'grad_row_mean_std', 'grad_row_mean_max',
                             'grad_sparsity', 'grad_top10_row_eccentricity', 'grad_top10_col_eccentricity']
        selected_layers = [str(layer) for layer in range(0,32)]
    elif model_name == "gpt-neo-2.7B":
        selected_modules = ["c_proj", "c_fc", 'q_proj', 'k_proj', 'v_proj', 'out_proj']
        selected_features = ['grad_abs_mean', 'grad_std',
                             'grad_top10p_ratio', 'grad_row_mean_std', 'grad_row_mean_max',
                             'grad_sparsity', 'grad_top10_row_eccentricity', 'grad_top10_col_eccentricity']
    elif model_name == "gpt-j-6b":
        selected_modules = ["fc_in", "fc_out", 'q_proj', 'k_proj', 'v_proj', 'out_proj']
        selected_features = ['grad_abs_mean', 'grad_std',
                             'grad_top10p_ratio', 'grad_row_mean_std', 'grad_row_mean_max',
                             'grad_sparsity', 'grad_top10_row_eccentricity', 'grad_top10_col_eccentricity']
    elif model_name == "llama-7b" or model_name == "llama-13b":
        selected_modules = [
            "down_proj",
            "up_proj",
            "gate_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]
        selected_features = [
            'grad_abs_mean',
            'grad_std',
            'grad_top10p_ratio',
            'grad_sparsity',
            'grad_top10_row_eccentricity',
            'grad_top10_col_eccentricity',
            'grad_row_mean_std',
            'grad_row_mean_max',
        ]
    else:
        selected_modules = [
            "down_proj",
            "up_proj",
            "gate_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]
    if selected_features is None:
        selected_features = ['grad_abs_mean', 'grad_mean', 'grad_std',
                             'grad_l2', 'grad_top10p_ratio', 'grad_row_mean_std', 'grad_row_mean_max',
                             'grad_sparsity']
    if selected_layers is None:
        selected_layers = [str(layer) for layer in range(0, 32)]
    if selected_modules is None:
        selected_modules = ["down_proj", "up_proj", "gate_proj", "q_proj", "k_proj", "v_proj", "o_proj"]

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        index = data['metadata']['sample_info']['sample_index']
        label = data['metadata']['sample_info']['sample_label']

        epoch_data = data['gradient_update_history'][epoch]
        layers = epoch_data['layer_updates']

        output_update = None

        features = []
        target_feature = 'avg_update'

        if "null" in layers:
            if 'embedding' not in layers["null"]:
                output_update, layernorm_update, embedding_update = (
                    layers["null"]['output']['avg_update'],
                    layers["null"]['layernorm']['avg_update'],
                    0
                )
                for sub_module in ['output', 'layernorm', 'embedding']:
                    for submodule in layers["null"][sub_module]['submodule_details']:
                        for feat in submodule:
                            if feat not in ['param_name', 'submodule_name',
                                            'grad_mean'] and 'amount' not in feat and 'min' not in feat:
                                features.append(layers["null"][sub_module]['submodule_details'][0][feat])
            else:
                if 'output' not in layers["null"]:
                    os.remove(json_path)
                    return None, None, None
                output_update, layernorm_update, embedding_update = (
                    layers["null"]['output']['avg_update_abs'],
                    layers["null"]['layernorm']['avg_update_abs'],
                    layers["null"]['embedding']['avg_update_abs']
                )
                for sub_module in ['output', 'layernorm', 'embedding']:
                    for submodule in layers["null"][sub_module]['submodule_details']:
                        for feat in submodule:
                            if feat not in ['param_name', 'submodule_name'] and 'amount' not in feat and 'min' not in feat:
                                features.append(layers["null"][sub_module]['submodule_details'][0][feat])
                    features.append(layers["null"][sub_module]['avg_update_abs'])

            del layers["null"]

        for layer_idx in sorted(layers.keys(), key=lambda x: int(x)):
            if selected_layers is not None and layer_idx not in selected_layers:
                continue

            layer = layers[layer_idx]
            ffn_name = 'ffn'
            attention_name = 'attention'
            if train_mode == "lora":
                for sub_module in layer[ffn_name]['submodule_details']:
                    if 'lora_A' in sub_module['param_name']:
                        continue
                    if selected_modules is not None:
                        module_name = sub_module['param_name'].split('.')[-4]
                        if module_name not in selected_modules:
                            continue
                    for feat in selected_features:
                        if feat in sub_module:
                            if feat_trans:
                                if 'max' in feat:
                                    trans_feat = np.sqrt(np.abs(sub_module[feat])) * np.sign(sub_module[feat])
                                elif 'min' in feat:
                                    trans_feat = np.power(sub_module[feat], 2)
                                else:
                                    trans_feat = sub_module[feat]
                                features.append(trans_feat)
                            else:
                                features.append(sub_module[feat])

                for sub_module in layer[attention_name]['submodule_details']:
                    if 'lora_A' in sub_module['param_name']:
                        continue
                    if selected_modules is not None:
                        module_name = sub_module['param_name'].split('.')[-4]
                        if module_name not in selected_modules:
                            continue
                    for feat in selected_features:
                        if feat in sub_module:
                            features.append(sub_module[feat])
            else:
                for sub_module in layer['ffn']['submodule_details']:
                    for feat in sub_module:
                        if feat not in ['param_name', 'submodule_name'] and 'amount' not in feat and 'min' not in feat:
                            features.append(sub_module[feat])

                for sub_module in layer['attention']['submodule_details']:
                    for feat in sub_module:
                        if feat not in ['param_name', 'submodule_name'] and 'amount' not in feat and 'min' not in feat:
                            features.append(sub_module[feat])
                for sub_module in layer['other']['submodule_details']:
                    for feat in sub_module:
                        if feat not in ['param_name', 'submodule_name'] and 'amount' not in feat and 'min' not in feat:
                            features.append(sub_module[feat])

        features_np = np.array(features)
        features_np[np.isinf(features_np)] = 0.01
        features_np[np.isnan(features_np)] = 0

        return features_np, label, index

    except Exception as e:
        return None, None, None


def load_all_jsons(json_dir, save_path=None, epoch=0, train_mode="lora", model_name="opt-6.7b"):
    json_paths = [
        os.path.join(json_dir, f)
        for f in os.listdir(json_dir)
        if f.endswith('.json') and f.startswith('sample_')
    ]
    if len(json_paths) == 0:
        raise ValueError(f"No matching JSON files found in {json_dir} (naming format: sample_*.json)")

    X_list = []
    y_list = []
    z_list = []
    num_limit = 0
    for path in json_paths:
        if num_limit >= 100000:
            break

        X, y, z = load_single_json_upgrade(path, epoch, train_mode, model_name=model_name)
        if X is not None and y is not None:
            X_list.append(X)
            y_list.append(y)
            z_list.append(z)
            num_limit += 1

    X = np.array(X_list)
    y = np.array(y_list)
    Z = np.array(z_list)
    X, y, Z = shuffle(X, y, Z, random_state=42)

    X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(
        X, y, Z, test_size=0.7, random_state=42, stratify=y
    )

    dataset = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "Z_train": Z_train,
        "Z_test": Z_test,
        "feature_names": [f"layer_{i // 2}_{'ffn' if i % 2 == 0 else 'attention'}" for i in range(64)],
        "metadata": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_dim": X_train.shape[1],
            "classes": np.unique(y).tolist(),
            "class_distribution": np.bincount(y).tolist()
        }
    }

    if save_path:
        save_path = os.path.join(save_path, f"processed_dataset_{epoch}.npz")
        save_dataset(dataset, save_path)

    return dataset


def save_dataset(dataset, save_path):
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    file_ext = os.path.splitext(save_path)[1].lower()

    if file_ext == '.npz':
        np.savez_compressed(
            save_path,
            X_train=dataset["X_train"],
            X_test=dataset["X_test"],
            y_train=dataset["y_train"],
            y_test=dataset["y_test"],
            Z_train=dataset["Z_train"],
            Z_test=dataset["Z_test"],
            feature_names=np.array(dataset["feature_names"], dtype=object),
            metadata=dataset["metadata"]
        )


def load_saved_dataset(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.npz':
        data = np.load(file_path, allow_pickle=True)
        dataset = {
            "X_train": data["X_train"],
            "X_test": data["X_test"],
            "y_train": data["y_train"],
            "y_test": data["y_test"],
            "Z_train": data["Z_train"],
            "Z_test": data["Z_test"],
            "feature_names": data["feature_names"].tolist(),
            "metadata": dict(data["metadata"].item()) if "metadata" in data else {}
        }

    return dataset


if __name__ == "__main__":
    train_mode = "lora"
    train_type = "lora"
    dataset_name = "WikiMIA"
    subset = "pubmed_central"
    model_type = "llama-7b"

    subsets = ["pile_cc", "arxiv", "dm_mathematics", "hackernews", "github", "pubmed_central", "wikipedia_(en)", "c4"]
    model_names = ["pythia-6.9b", "opt-6.7b", "llama-7b", "gpt-j-6b", "gpt-neo-2.7B", "llama-13b"]
    dataset_names = ["WikiMIA", "arXivTection", "WikiMIA_unify", "WikiMIA_remove", "BookTection", "BookMIA", "ArxivMIA", "mimir"]

    seed = 99

    if dataset_name != "mimir" and dataset_name != "pile":
        json_directory = f"{DATA_CONTAIN_BASE}/{model_type}/seed_{seed}/{dataset_name}/{train_type}/"
        save_file_path = f"{DATA_CONTAIN_BASE}/{model_type}/seed_{seed}/{dataset_name}/training_data_{train_type}"
    else:
        json_directory = f"{DATA_CONTAIN_BASE}/{model_type}/seed_{seed}/{dataset_name}/{subset}/{train_type}/"
        save_file_path = f"{DATA_CONTAIN_BASE}/{model_type}/seed_{seed}/{dataset_name}/{subset}/training_data_{train_type}"