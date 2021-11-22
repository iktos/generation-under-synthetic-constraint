"""
Copyright (C) Iktos - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

import argparse
import os

from guacamol.guacamol.assess_goal_directed_generation import (
    assess_goal_directed_generation,
)
from guacamol.guacamol.utils.helpers import setup_default_logger
from .smiles_rnn_directed_generator import SmilesRnnDirectedGenerator


def goal_directed_generation(
        suite,
        synth_score,
        n_epochs,
        mols_to_sample: int = 1024,
        keep_top: int = 512,
        optimize_n_epochs: int = 2,
        max_len: int = 100,
        optimize_batch_size: int = 256,
        number_final_samples: int = 0,
        random_start: bool = False,
        n_jobs: int = -1,
        seed: int = 42,
        output_file: str = None,
        suffix_coll="",
        smiles_file: str = None,
        pretrained_model_path: str = None,
):
    if not os.getenv('MONGO_URL'):
        raise Exception("No MONGO_URL environment variable was set")

    if not os.getenv("DB_STORAGE"):
        raise Exception("No DB_STORAGE environment variable was set")

    if "pi3kmtor" in suite:
        if pretrained_model_path is None:
            pretrained_model_path = "guacamol_baselines/smiles_lstm_hc/pretrained_model/model_final_pi3kmtor.pt"
        if smiles_file is None:
            smiles_file = "pi3kmtor/pi3kmtor.smiles"

    if pretrained_model_path is None:
        pretrained_model_path = "guacamol_baselines/smiles_lstm_hc/pretrained_model/model_final_chembl.pt"

    if smiles_file is None:
        smiles_file = "guacamol_baselines/data/guacamol_v1_all.smiles"

    if output_file is None:
        output_file =  "guacamol_baselines/smiles_lstm_hc/results/goal_directed_results_" + suite+ ".json"

    optimizer = SmilesRnnDirectedGenerator(
        pretrained_model_path=pretrained_model_path,
        n_epochs=n_epochs,
        mols_to_sample=mols_to_sample,
        keep_top=keep_top,
        optimize_n_epochs=optimize_n_epochs,
        max_len=max_len,
        optimize_batch_size=optimize_batch_size,
        number_final_samples=number_final_samples,
        random_start=random_start,
        smi_file=smiles_file,
        n_jobs=n_jobs,
        seed=seed,
        synth_score=synth_score,
        suffix_coll=suffix_coll,
    )

    assess_goal_directed_generation(
        optimizer, json_output_file=output_file, benchmark_version=suite,
        synth_score=synth_score
    )



if __name__ == "__main__":
    setup_default_logger()

    parser = argparse.ArgumentParser(
        description="Goal-directed generation benchmark for SMILES RNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Full path to the pre-trained SMILES RNN model",
    )
    parser.add_argument(
        "--max_len", default=100, type=int, help="Max length of a SMILES string"
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument(
        "--output_dir", default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--number_repetitions",
        default=1,
        type=int,
        help="Number of re-training runs to average",
    )
    parser.add_argument(
        "--keep_top", default=512, type=int, help="Molecules kept each step"
    )
    parser.add_argument("--n_epochs", default=20, type=int, help="Epochs to sample")
    parser.add_argument(
        "--mols_to_sample",
        default=1024,
        type=int,
        help="Molecules sampled at each step",
    )
    parser.add_argument(
        "--optimize_batch_size",
        default=256,
        type=int,
        help="Batch size for the optimization",
    )
    parser.add_argument(
        "--optimize_n_epochs",
        default=2,
        type=int,
        help="Number of epochs for the optimization",
    )
    parser.add_argument(
        "--benchmark_num_samples",
        default=0,
        type=int,
        help="Number of molecules to generate from final model for the benchmark",
    )
    parser.add_argument(
        "--benchmark_trajectory",
        action="store_true",
        help="Take molecules generated during re-training into account for the benchmark",
    )
    parser.add_argument(
        "--smiles_file", default="guacamol_baselines/data/guacamol_v1_all.smiles"
    )
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--suite", default="pi3kmtor")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--synth_score", default=None)
    parser.add_argument("--suffix_coll", default="")

    args = parser.parse_args()

    if not os.getenv('MONGO_URL'):
        raise Exception("No MONGO_URL environment variable was set")

    if not os.getenv("DB_STORAGE"):
        raise Exception("No DB_STORAGE environment variable was set")

    if "pi3kmtor" in args.suite:
        args.model_path = (
            "guacamol_baselines/smiles_lstm_hc/pretrained_model/model_final_pi3kmtor.pt"
        )
        args.smiles_file = "pi3kmtor/pi3kmtor.smiles"

    else:
        args.model_path = "guacamol_baselines/smiles_lstm_hc/pretrained_model/model_final_chembl.pt"

    if args.output_file is None:
        json_file_path = (
            "guacamol_baselines/smiles_lstm_hc/results/goal_directed_results_"
            + args.suite
            + ".json"
        )
        args.output_file = json_file_path


    goal_directed_generation(suite=args.suite,
                             synth_score=args.synth_score,
                             n_epochs=args.n_epochs,
                             mols_to_sample=args.mols_to_sample,
                             keep_top=args.keep_top,
                             optimize_n_epochs = args.optimize_n_epochs,
                             max_len=args.max_len,
                             optimize_batch_size=args.optimize_batch_size,
                             random_start=args.random_start,
                             n_jobs= args.n_jobs,
                             seed=args.seed,
                             output_file=args.output_file,
                             suffix_coll=args.suffix_coll,
                             smiles_file=args.smiles_file,
                             pretrained_model_path=args.model_path,
                             )
