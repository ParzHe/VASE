import torch
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
from sklearn.metrics import auc


def compute_aug(hall_scores, hall_labels, resolution=100):
    """
    Compute the AUG.

    Parameters:
        hall_scores (list or np.ndarray):
        hall_labels (list or np.ndarray): 
        resolution (int): Number of cut-off percentages to evaluate (default is 100).

    Returns:
        float: The computed AUG value.
    """
    # Ensure inputs are numpy arrays
    hall_scores = np.array(hall_scores)
    hall_labels = np.array(hall_labels)

    # Sort samples by hallucination scores in ascending order
    sorted_indices = np.argsort(hall_scores)
    sorted_labels = hall_labels[sorted_indices]

    # Initialize variables for AURAC computation
    total_samples = len(hall_scores)
    cut_offs = np.linspace(0, 1, resolution + 1)  # Cut-off percentages (0% to 100%)
    green_list = []

    for cut_off in cut_offs:
        # Number of samples in the confident subset
        num_selected = int(cut_off * total_samples)
        
        # Compute accuracy for the confident subset
        if num_selected > 0:
            confident_subset = sorted_labels[:num_selected]
            green_mean = np.mean(confident_subset)
        else:
            green_mean = 0
        green_list.append(green_mean)

    aug = auc(cut_offs, green_list)
    return aug


def get_auc_aug(csv_hall_label, column_hall_label,
                csv_hall_score, column_hall_score,
                uncertainty_flag):
    hall_label_df = pd.read_csv(csv_hall_label, usecols=[column_hall_label])
    hall_score_df = pd.read_csv(csv_hall_score, usecols=[column_hall_score])

    hall_label_series = pd.to_numeric(hall_label_df[column_hall_label], errors='coerce')
    hall_score_series = pd.to_numeric(hall_score_df[column_hall_score], errors='coerce')

    # Align row counts and drop invalid rows so sklearn never sees NaN/Inf.
    min_len = min(len(hall_label_series), len(hall_score_series))
    eval_df = pd.DataFrame({
        'hall_label': hall_label_series.iloc[:min_len].to_numpy(),
        'hall_score': hall_score_series.iloc[:min_len].to_numpy(),
    })
    eval_df = eval_df.replace([np.inf, -np.inf], np.nan).dropna()
    if eval_df.empty:
        raise ValueError('No valid rows remain after dropping NaN/Inf values.')

    hall_label = torch.tensor(eval_df['hall_label'].to_numpy(), dtype=torch.float32)
    hall_label_int = (hall_label < 1.0).long() # green<1: hallucination, green==1: non-hallucination

    hall_score = torch.tensor(eval_df['hall_score'].to_numpy(), dtype=torch.float32)
    score_min, score_max = hall_score.min(), hall_score.max()
    if torch.isclose(score_max, score_min):
        hall_score_norm = torch.zeros_like(hall_score)
    else:
        hall_score_norm = (hall_score - score_min) / (score_max - score_min)

    if uncertainty_flag == False:
        hall_score_norm = 1.0 - hall_score_norm

    # AUC requires both classes to be present.
    if torch.unique(hall_label_int).numel() < 2:
        raise ValueError('AUC is undefined because only one class is present in hall labels after filtering.')
    auc_score = metrics.roc_auc_score(hall_label_int.cpu().numpy(), hall_score_norm.cpu().numpy())

    # AUG
    aug_score = compute_aug(hall_score_norm.cpu().numpy(), hall_label.cpu().numpy())
    return round(auc_score * 100, 2), round(aug_score * 100, 2)


def main(
    csv_hall_label="outputs/radvqa_medgemma_green.csv",
    column_hall_label="green_score",
    csv_hall_score="outputs/radvqa_medgemma_hallscore.csv",
    column_hall_score="SE", # SE, VASE, RadFlag
    uncertainty_flag=True, # True when hall_score indicates uncertainty (SE, VASE), False when hall_score indicates confidence (RadFlag).
):
    auc, aug = get_auc_aug(csv_hall_label, column_hall_label, 
                           csv_hall_score, column_hall_score, 
                           uncertainty_flag)
    print('AUC:', auc, '%', 'AUG:', aug, '%')
    return auc, aug


# python hall_det_eval.py
if __name__ == "__main__":
    main() 
    
