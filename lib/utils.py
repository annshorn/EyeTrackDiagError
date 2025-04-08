import argparse
import yaml
import os
from datetime import datetime
import pickle
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    accuracy_score, f1_score
)

def parse_args():
    """Parses command-line arguments for the experiment."""
    parser = argparse.ArgumentParser(description="Parse configuration file for experiment setup.")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the YAML configuration file."
    )

    args = parser.parse_args()

    # Validate config file existence
    if not os.path.isfile(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    return args

def load_config(config_path):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
            return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

def create_save_folder(model_prefix, **kwargs):
    """Creates a timestamped directory using values from kwargs."""
    output_folder = kwargs.get("output_folder", "./")
    output_name = kwargs.get("output_name", "default_name")

    save_model_name = f"{model_prefix}_{output_name}"
    save_model_path = os.path.join(output_folder, f"{save_model_name}_{datetime.now().strftime('%Y-%m-%d___%H-%M-%S')}")

    os.makedirs(save_model_path, exist_ok=True)
    print(f"Model will be saved in: {save_model_path}")
    return save_model_path

def load_pickle_file(file_path):
    """Load a grouped Pandas DataFrame from a pickle file."""
    with open(file_path, 'rb') as pickle_file:
        dataframe = pickle.load(pickle_file)
    return dataframe

def save_dicts(my_dict, json_file_name):
    """Load a grouped Pandas DataFrame into a pickle file."""
    with open(json_file_name+'.pkl', 'wb') as json_file:
        pickle.dump(my_dict, json_file)


def model_eval(
        model: torch.nn.Module,
        weights: dict,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device
    ):
    """
    Evaluate the given model on the test set and print various performance metrics.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        weights (dict): Model weights (state_dict).
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the evaluation on (CPU/GPU).
    """
    try:
        model.load_state_dict(weights)
    except:
        model.load_state_dict(weights["state_dict"])

    model.to(device)
    model.eval()

    labels_validation = []
    pred = []

    for batch in test_loader:
        with torch.no_grad():
            if len(batch) == 2:
                seq, labels = batch
                y_pred = model(seq.to(device))
            elif len(batch) == 3:
                seq, img, labels = batch
                y_pred =  model(seq.to(device), img.to(device))
            pred.extend(y_pred.cpu().numpy())
            labels_validation.extend(labels.cpu().numpy())
    pred = np.array(pred)
    labels_validation = np.array(labels_validation)
    
    # Compute ROC AUC Score
    roc_auc = roc_auc_score(labels_validation, pred)
    print(f'ROC AUC Score: {roc_auc:.4f}')

    # Compute ROC Curve
    fpr, tpr, thresholds = roc_curve(labels_validation, pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    # Apply optimal threshold
    prediction = (pred > optimal_threshold).astype(int)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(labels_validation, prediction)
    print("\nConfusion Matrix:\n", conf_matrix)

    # Compute classification report
    print("\nClassification Report:\n", classification_report(labels_validation, prediction, digits=4))

    # Compute Sensitivity (Recall) and Specificity
    tn, fp, fn, tp = conf_matrix.ravel()
    recall = tp / (tp + fn)  # Sensitivity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Print additional metrics
    print(f'Accuracy:  {accuracy_score(labels_validation, prediction):.4f}')
    print(f'F1 Score:  {f1_score(labels_validation, prediction):.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall (Sensitivity): {recall:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print('=============')
        
