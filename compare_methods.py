import argparse

from hall_det_eval import get_auc_aug


# method name -> uncertainty_flag (True = higher score means more uncertain / more hallucinated)
METHODS = {
    "SE": True,
    "VASE": True,
    "UniVRSE": True,
    "RadFlag": False,
}


def main():
    parser = argparse.ArgumentParser(
        description="Print a unified AUC/AUG table for SE, VASE, RadFlag, UniVRSE.",
    )
    parser.add_argument(
        "--score_csv",
        default="outputs/radvqa_medgemma_hallscore.csv",
        help="CSV produced by VASE/main_univrse_det.py.",
    )
    parser.add_argument(
        "--label_csv",
        default="outputs/radvqa_medgemma_green.csv",
        help="CSV produced by VASE/green_eval.py on the score_csv above.",
    )
    parser.add_argument(
        "--label_column", default="green_score",
        help="Column holding the GREEN score in label_csv.",
    )
    args = parser.parse_args()

    rows = []
    for method, uncertainty_flag in METHODS.items():
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