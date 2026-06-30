import torch
import sys
import os
import time
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    ToTensor,
    Compose,
    ColorJitter,
    RandomResizedCrop,
    RandomHorizontalFlip,
    Normalize,
    Resize,
)
import torch.optim as optim
import copy
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random
import numpy as np
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import math
from torch.utils.data import Subset, DataLoader
from collections import defaultdict
import random

import torch                  # For tensor operations, model handling
import numpy as np            # For numerical operations (used in weight reshaping)
import pandas as pd           # To handle DataFrame operations (if df is a DataFrame)

def log_and_print(message, log_file="deepu_log.txt"):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

def evaluate_model(model, test_loader, device):
    """
    Evaluate a model and calculate Top-1 and Top-5 accuracies.
    Handles models that return tuples as outputs.
    """
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculations for evaluation
        for batch_idx, batch_data in enumerate(test_loader):
            # Debugging: Inspect batch data
            # print(f"Batch {batch_idx}: {type(batch_data)}, Length: {len(batch_data)}")
            
            # Unpack the data
            if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
                inputs, labels = batch_data[:2]  # Extract the first two elements (inputs and labels)
            else:
                raise ValueError(f"Unexpected data format in batch {batch_idx}: {batch_data}")

            # Move inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Extract logits if outputs is a tuple
            if isinstance(outputs, tuple):
                logits = outputs[0]  # Get the first component (logits)
            else:
                logits = outputs  # If not a tuple, use outputs directly

            # Calculate Top-1 and Top-5 accuracies
            _, predictions = logits.topk(5, dim=1, largest=True, sorted=True)
            total += labels.size(0)
            correct_top1 += (predictions[:, 0] == labels).sum().item()
            correct_top5 += sum([labels[i] in predictions[i] for i in range(labels.size(0))])

    top1_accuracy = correct_top1 / total
    top5_accuracy = correct_top5 / total
    # print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    # print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    return top1_accuracy, top5_accuracy


# start_time_total = time.time()
def compute_grad_influence_mapping(model, data_forget, data_retain, first_layer_name, exp_dir, device=None,
                                   max_batches=64, snr_low=0.7, snr_high=1.1):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    os.makedirs(exp_dir, exist_ok=True)

    def classify_weight(snr):
        # SNR below snr_low -> weight barely matters to the forget data.
        if snr < snr_low:
            return 'Non_Influential_Weight'
        # SNR between snr_low and snr_high -> weight matters to both forget and retain data.
        elif snr_low <= snr <= snr_high:
            return 'Shared_Weight'
        # SNR above snr_high -> weight is strongly tied to the forget data.
        else:
            return 'Influential_Weight'


    file_path = os.path.join(exp_dir, f'{first_layer_name.replace(".", "_")}_weights_snr.pt')
    if os.path.exists(file_path):
        print(f"DataFrame already computed for layer '{first_layer_name}'. Loading from file.")
        df = torch.load(file_path, weights_only=False)
        return df

    def calculate_grads(model, data_loader, first_layer_name, device, max_batches=5, cache_dir=None, tag=None):
        cache_path = None
        if cache_dir and tag:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{first_layer_name.replace('.', '_')}_{tag}_grads.pt")
            if os.path.exists(cache_path):
                print(f"Loading cached gradients for {tag} from: {cache_path}")
                cached = torch.load(cache_path, weights_only=False)
                return cached["params"].to(device), cached["grads"].to(device)

        num_params = model.state_dict()[first_layer_name].numel()
        print(f"Computing gradients for {first_layer_name} on {tag} (max {max_batches} batches)...")
        layer_grads = torch.zeros(num_params, device=device)
        params = None

        for batch_idx, (images, labels) in enumerate(tqdm(data_loader)):
            if batch_idx >= max_batches:
                break

            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()

            for name, param in model.named_parameters():
                if name == first_layer_name and param.grad is not None:
                    if params is None:
                        params = param.flatten().detach().clone()
                    grads = param.grad.flatten().detach()
                    layer_grads += grads.abs()

        layer_grads /= (batch_idx + 1)

        if cache_path:
            torch.save({"params": params.cpu(), "grads": layer_grads.cpu()}, cache_path)

        return params, layer_grads

    params, grads_forget = calculate_grads(model, data_forget, first_layer_name, device,
                                        max_batches=max_batches, cache_dir=exp_dir, tag="forget")
    _, grads_retain = calculate_grads(model, data_retain, first_layer_name, device,
                                    max_batches=max_batches, cache_dir=exp_dir, tag="retain")

    # Create a dataframe to store results
    df = pd.DataFrame({'Weight_Index': np.arange(len(params.cpu().numpy())),
                       'Weight_Value': params.cpu().numpy(),
                       'Grad_Forget': grads_forget.cpu().numpy(),
                       'Grad_Retain': grads_retain.cpu().numpy()})


    df['SNR'] = (df['Grad_Forget']**2) / (df['Grad_Retain']**2 + 1e-12)

        # Handle infinite and NaN values
    df['SNR'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['SNR'], inplace=True)

    # Apply the classification function to assign labels
    df['Cluster_Label'] = df['SNR'].apply(classify_weight)


        # Count total number of weights per category
    total_weights = len(df)
    total_non_influential = (df['Cluster_Label'] == 'Non_Influential_Weight').sum()
    total_shared = (df['Cluster_Label'] == 'Shared_Weight').sum()
    total_influential = (df['Cluster_Label'] == 'Influential_Weight').sum()
    
    print(f"Total Weights: {total_weights}")
    print(f"Total Non-Influential Weights: {total_non_influential}")
    print(f"Total Shared Weights: {total_shared}")
    print(f"Total Influential Weights: {total_influential}")

    df = df[['Weight_Index', 'Weight_Value', 'Grad_Forget', 'Grad_Retain', 'SNR', 'Cluster_Label']]

    torch.save(df, file_path)
    print(f"Updated DataFrame saved at: {file_path}")

    return 


def update_weights_with_kmeans(layer_name, mapping_folder,percentile_threshold_influential,percentile_threshold_shared,noise_scale=1.0, decay_factor=0.1):
    import os
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans

    # Ensure mapping folder exists
    os.makedirs(mapping_folder, exist_ok=True)

    # Construct the file path for the layer's weight-SNR file
    file_path = os.path.join(mapping_folder, f"{layer_name.replace('.', '_')}_weights_snr.pt")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Load the dataframe from file and keep relevant columns
    df = torch.load(file_path, weights_only=False)
    required_columns = ["Weight_Index", "Weight_Value", "Grad_Forget", "Grad_Retain", "SNR", "Cluster_Label"]
    df = df[required_columns]
    
    # Create a copy of df to update
    df_updated = df.copy()

    # ===== Update Influential Weights (far from centroid) =====
    df_influential = df[df["Cluster_Label"] == "Influential_Weight"].copy()
    if not df_influential.empty and len(df_influential) >= 2:
        # Apply KMeans clustering with 2 clusters on "Weight_Value" and "SNR"
        X_influential = df_influential[["Weight_Value", "SNR"]].values
        kmeans_influential = KMeans(n_clusters=2, random_state=42, n_init=10)
        df_influential["Cluster"] = kmeans_influential.fit_predict(X_influential)
        centroids_influential = kmeans_influential.cluster_centers_
        
        # Compute Euclidean distance of each influential weight to its assigned centroid
        df_influential["Distance_to_Centroid"] = np.linalg.norm(
            X_influential - centroids_influential[df_influential["Cluster"]],
            axis=1
        )
        
        # Determine the distance threshold based on the given percentile for influential weights
        threshold_influential = np.percentile(df_influential["Distance_to_Centroid"], percentile_threshold_influential)
        print(f"Influential Weights - Distance threshold (at {percentile_threshold_influential}th percentile): {threshold_influential}")

        # Compute noise based on the standard deviation of Grad_Forget
        noise_std = df_influential["Grad_Forget"].std()
        noise_vector = np.random.normal(0, noise_std * 1.0, size=len(df_influential))

        # Assign each row a noise value
        df_influential["Noise"] = noise_vector

        def update_influential_weight(row):
            if row["Distance_to_Centroid"] > threshold_influential:
                return 0.0
            else:
                return row["Weight_Value"] + row["Noise"]

        df_influential["Updated_Weight"] = df_influential.apply(update_influential_weight, axis=1)
        df_updated.loc[df_influential.index, "Updated_Weight"] = df_influential["Updated_Weight"]

    # ===== Update Shared Weights (close to centroid) =====
    df_shared = df[df["Cluster_Label"] == "Shared_Weight"].copy()
    if not df_shared.empty:
        X_shared = df_shared[["Weight_Value", "SNR"]].values
        if X_shared.shape[0] >= 2:
            kmeans_shared = KMeans(n_clusters=2, random_state=42, n_init=10)
            df_shared["Cluster"] = kmeans_shared.fit_predict(X_shared)
            centroids_shared = kmeans_shared.cluster_centers_

            # Override noise_scale for shared weights
            # noise_scale = 0.6
            noise_std = df_shared["Grad_Forget"].std()
            noise_vector = np.random.normal(0, noise_std * noise_scale, size=len(df_shared))
            df_shared["Noise"] = noise_vector

            # Compute Euclidean distance of each shared weight to its assigned centroid
            df_shared["Distance_to_Centroid"] = np.linalg.norm(
                X_shared - centroids_shared[df_shared["Cluster"].values],
                axis=1
            )
            
            # Determine the distance threshold based on the given percentile for shared weights
            threshold_shared = np.percentile(df_shared["Distance_to_Centroid"], percentile_threshold_shared)
            print(f"Shared Weights - Distance threshold (at {percentile_threshold_shared}th percentile): {threshold_shared}")

            # For shared weights, apply weight decay if they are close to the centroid.
            def update_shared_weight(row):
                if row.get("Distance_to_Centroid", float('inf')) <= threshold_shared:
                    return row["Weight_Value"] * np.exp(-decay_factor * abs(row["Grad_Forget"]))
                else:
                    return row["Weight_Value"] + row["Noise"]

            df_shared["Updated_Weight"] = df_shared.apply(update_shared_weight, axis=1)
        else:
            # Not enough shared weights for clustering; keep original weights.
            print("Not enough shared weights to perform KMeans clustering. Keeping original weights.")
            df_shared["Updated_Weight"] = df_shared["Weight_Value"]

        df_updated.loc[df_shared.index, "Updated_Weight"] = df_shared["Updated_Weight"]

    # ===== For Non-Influential/Non-Shared Weights, keep original weights =====
    df_updated.loc[~df_updated["Cluster_Label"].isin(["Influential_Weight", "Shared_Weight"]), "Updated_Weight"] = df_updated["Weight_Value"]

    # Save the updated dataframe to the same file
    updated_file_path = os.path.join(mapping_folder, f"{layer_name.replace('.', '_')}_weights_snr.pt")
    torch.save(df_updated, updated_file_path)
    print(f"Updated DataFrame saved at: {updated_file_path}")

    return 

def update_model(model, layers_list, mapping_folder, retain_loader, forget_loader, test_loader, device,
                 learning_rate=0.001,bp_every_n_layers = 1, log_file="deepu_log2.txt"):

    updated_model = copy.deepcopy(model).to(device)
    updated_model.eval()

    optimizer = optim.SGD(updated_model.parameters(), lr=learning_rate, momentum=0.9)

    layer_counter = 0
    eval_counter = 0
    total_layers = len(layers_list)
    log_and_print(f"Total layers to update: {total_layers}", log_file)
    updated_any = False  # tracks if any layer was updated in the current block

    for i, layer_name in enumerate(tqdm(layers_list, desc="Updating layers", unit="layer")):
        file_path = os.path.join(mapping_folder, f"{layer_name.replace('.', '_')}_weights_snr.pt")
        if not os.path.exists(file_path):
            log_and_print(f"Mapping file not found for layer {layer_name}. Skipping.", log_file)
            continue

        df = torch.load(file_path, weights_only=False)
        if "Updated_Weight" not in df.columns:
            log_and_print(f"'Updated_Weight' column not found in {file_path}. Skipping.", log_file)
            continue

        changed_mask = df["Updated_Weight"] != df["Weight_Value"]
        changed_indices = np.where(changed_mask.values)[0]
        num_changed = len(changed_indices)

        if num_changed == 0:
            log_and_print(f"No changed weights found for layer {layer_name}. Skipping.", log_file)
            continue

        updated_weights = torch.tensor(df["Updated_Weight"].values, dtype=torch.float32).to(device)
        module = dict(updated_model.named_parameters()).get(layer_name)
        if module is None:
            log_and_print(f"Layer {layer_name} not found in model. Skipping.", log_file)
            continue

        current_weights = module.data.flatten()
        current_weights[changed_indices] = updated_weights[changed_indices]
        module.data = current_weights.view(module.data.shape)

        log_and_print(f"Layer {layer_name}: Updated {num_changed} weights.", log_file)

        layer_counter += 1
        eval_counter += 1
        updated_any = True

        if updated_any and ((layer_counter % bp_every_n_layers == 0) or (i == total_layers - 1)):
        # if layer_counter % bp_every_n_layers == 0 or i == total_layers - 1:
            # print(f"Backpropagating after updating {layer_counter} layers using {len(subset_indices)} samples...")
            updated_model.train()
            running_loss = 0.0
            max_batches = 64
            for b_idx, (data, target) in enumerate(tqdm(retain_loader, desc="Backprop batches", leave=False)):
                # if b_idx >= max_batches:
                #     break
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = updated_model(data)
                if isinstance(output, tuple):
                    output = output[0]
                loss = torch.nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            log_and_print(f"Loss after backprop (after {layer_counter} layers): {running_loss / max_batches:.4f}", log_file)
            updated_model.eval()
            updated_any = False

        eval_every_n_layers = 3  

        if eval_counter >= eval_every_n_layers:
            top1, _ = evaluate_model(updated_model, test_loader, device)
            log_and_print(f"Top-1 Accuracy after {layer_counter} layers updated: {top1:.4f}", log_file)
            eval_counter = 0

    return updated_model




def get_layers(model, first_n=20, last_n=15):
    param_names = list(dict(model.named_parameters()).keys())
    first = param_names[:first_n]
    last = param_names[-last_n:]
    return first + last  # You can change last_n if you want more or fewer layers from the end