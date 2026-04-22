"""Experiment-config loader shared by main_hall_det.py, main_univrse_det.py,
green_eval.py and compare_methods.py.

A config is a YAML file (see configs/) with (a subset of) these keys:

    name           : short identifier (for documentation only)
    model_id       : HF VLM id  (used by the sampling scripts)
    dataset        : HF dataset id  (used by the sampling scripts)
    question_set   : "open-ended" or "all"  (used by the sampling scripts)
    hallscore_csv  : path to the CSV of generated answers + SE/VASE/RadFlag[/UniVRSE]
    green_csv      : path to the GREEN-labeled CSV

Each script picks up the fields it needs; CLI flags always override the YAML.
"""
import argparse
import os

import yaml


def load_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def peek_config(argv=None):
    """Pre-parse argv for --config so the main parser can use it as defaults."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config", default=os.getenv("EXP_CONFIG"))
    ns, _ = peek.parse_known_args(argv)
    cfg = load_config(ns.config)
    cfg["__config_path__"] = ns.config
    return cfg
