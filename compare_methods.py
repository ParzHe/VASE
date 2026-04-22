import argparse

import pandas as pd

from hall_det_eval import get_auc_aug
from exp_config import peek_config


# method name -> uncertainty_flag (True = higher score means more uncertain / more hallucinated)
METHODS = {
    "SE": True,
    "VASE": True,
    "UniVRSE": True,
    "RadFlag": False,
}


def main():
    cfg = peek_config()
    parser = argparse.ArgumentParser(
        description="Print a unified AUC/AUG table for SE, VASE, RadFlag, UniVRSE. "
                    "Methods whose column is absent from the score CSV are silently skipped, "
                    "so this works for both main_hall_det.py and main_univrse_det.py outputs.",
    )
    parser.add_argument(
        "--config",
        default=cfg.get("__config_path__"),
        help="Path to a YAML experiment config (see configs/). CLI flags override config values.",
    )
    parser.add_argument(
        "--score_csv",
        default=cfg.get("hallscore_csv", "outputs/radvqa_medgemma_hallscore.csv"),
        help="CSV produced by main_hall_det.py or main_univrse_det.py.",
    )
    parser.add_argument(
        "--label_csv",
        default=cfg.get("green_csv", "outputs/radvqa_medgemma_green.csv"),
        help="CSV produced by green_eval.py on the score_csv above.",
    )
    parser.add_argument(
        "--label_column", default="green_score",
        help="Column holding the GREEN score in label_csv.",
    )
    args = parser.parse_args()

    available_columns = set(pd.read_csv(args.score_csv, nrows=0).columns)

    rows = []
    for method, uncertainty_flag in METHODS.items():
        if method not in available_columns:
            continue
        auc, aug = get_auc_aug(
            args.label_csv, args.label_column,
            args.score_csv, method,
            uncertainty_flag,
        )
        rows.append((method, auc, aug))

    print("| Method   | AUC(%) | AUG(%) |")
    print("|----------|--------|--------|")
    for method, auc, aug in rows:
        print(f"| {method:<8} | {auc:>6.2f} | {aug:>6.2f} |")


if __name__ == "__main__":
    main()
