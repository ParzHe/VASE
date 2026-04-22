import os
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from green_score import GREEN

from exp_config import peek_config


def eval_t2t_csv(in_csv_file, out_csv_file):
    model_name = "StanfordAIMI/GREEN-radllama2-7b"
    green_scorer = GREEN(model_name, output_dir=".")
    in_df = pd.read_csv(in_csv_file, usecols=[0, 1, 2, 3])

    in_df['ref_answer'] = in_df['question'] + " " + in_df['ref_answer']
    in_df['gen_answer'] = in_df['question'] + " " + in_df['gen_answer']
    refs_list = in_df['ref_answer'].tolist()
    hyps_list = in_df['gen_answer'].tolist()

    mean, std, green_score_list, summary, result_df = green_scorer(refs_list, hyps_list)

    result_df = result_df.drop(['reference', 'predictions'], axis=1)
    in_df.reset_index(drop=True, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    out_df = pd.concat([in_df, result_df], axis=1)

    os.makedirs(os.path.dirname(out_csv_file) or ".", exist_ok=True)
    out_df.to_csv(out_csv_file, index=False)


def parse_args():
    cfg = peek_config()
    parser = argparse.ArgumentParser(description="Evaluate GREEN scores for generated answers.")
    parser.add_argument(
        "--config",
        default=cfg.get("__config_path__"),
        help="Path to a YAML experiment config (see configs/). CLI flags override config values.",
    )
    parser.add_argument(
        "--input_csv", type=str,
        default=cfg.get("hallscore_csv", "outputs/radvqa_medgemma_hallscore.csv"),
        help="Path to the input CSV file containing questions, reference answers, and generated answers.",
    )
    parser.add_argument(
        "--output_csv", type=str,
        default=cfg.get("green_csv", "outputs/radvqa_medgemma_green.csv"),
        help="Path to the output CSV file where GREEN scores will be saved.",
    )
    return parser.parse_args()


# conda activate env_green (!!! Set up a dedicated virtual environment for the Green model to prevent conflicts with environments used for other MLLMs.)
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 green_eval.py
# deactivate
if __name__ == "__main__":
    args = parse_args()
    eval_t2t_csv(args.input_csv, args.output_csv)
