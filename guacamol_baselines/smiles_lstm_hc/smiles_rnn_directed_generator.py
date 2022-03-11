"""
Copyright (C) Iktos - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

from pathlib import Path
from typing import List, Optional

import joblib
import torch
from pymongo import MongoClient
import os
from guacamol.guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.guacamol.scoring_function import ScoringFunction
from guacamol.guacamol.utils.chemistry import canonicalize_list, canonicalize
from joblib import delayed

from .rnn_generator import SmilesRnnMoleculeGenerator
from .rnn_utils import load_rnn_model
from datetime import date
from exploit_results.add_scores_to_db import add_score_to_db_pi3kmtor


class SmilesRnnDirectedGenerator(GoalDirectedGenerator):
    def __init__(
        self,
        pretrained_model_path: str,
        n_epochs=4,
        mols_to_sample=1028,
        keep_top=512,
        optimize_n_epochs=2,
        max_len=100,
        optimize_batch_size=64,
        number_final_samples=1028,
        sample_final_model_only=False,
        random_start=False,
        smi_file=None,
        n_jobs=-1,
        seed=None,
        synth_score=None,
        suffix_coll=None,
    ) -> None:
        self.pretrained_model_path = pretrained_model_path
        self.n_epochs = n_epochs
        self.mols_to_sample = mols_to_sample
        self.keep_top = keep_top
        self.optimize_batch_size = optimize_batch_size
        self.optimize_n_epochs = optimize_n_epochs
        self.pretrain_n_epochs = 0
        self.max_len = max_len
        self.number_final_samples = number_final_samples
        self.sample_final_model_only = sample_final_model_only
        self.random_start = random_start
        self.smi_file = smi_file
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.client = MongoClient(os.getenv("MONGO_URL"))[os.getenv("DB_STORAGE")]
        self.seed = seed
        self.synth_score = synth_score
        self.suffix_coll = suffix_coll

    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            return self.pool(delayed(canonicalize)(s.strip()) for s in f)

    def top_k(self, smiles, scoring_function, k):
        joblist = (delayed(scoring_function.score)(s) for s in smiles)
        scores = self.pool(joblist)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        scoring_function_start: ScoringFunction,
        starting_population: Optional[List[str]] = None,
        benchmark_name=None,
    ) -> List[str]:

        # fetch initial population?
        if starting_population is None:
            print("selecting initial population...")
            if self.random_start:
                starting_population = []
            else:
                all_smiles = self.load_smiles_from_file(self.smi_file)
                print(len(all_smiles), " smiles loaded")

                starting_population = self.top_k(
                    all_smiles, scoring_function_start, self.mols_to_sample
                )

        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
        model_def = Path(self.pretrained_model_path).with_suffix(".json")

        model = load_rnn_model(
            model_def, self.pretrained_model_path, device, copy_to_cpu=True
        )

        generator = SmilesRnnMoleculeGenerator(
            model=model,
            max_len=self.max_len,
            device=device,
            name_benchmark=benchmark_name,
        )
        database_name = benchmark_name
        if len(self.suffix_coll) > 0:
            database_name = database_name + self.suffix_coll
        if self.synth_score:
            database_name = database_name  # +"_"+self.synth_score
        i = 0
        while database_name + "_" + str(i) in self.client.list_collection_names():
            i += 1

        database_name = database_name + "_" + str(i)

        print("Database in which molecules are stored : ", database_name)

        molecules = generator.optimise(
            objective=scoring_function,
            start_population=starting_population,
            n_epochs=self.n_epochs,
            mols_to_sample=self.mols_to_sample,
            keep_top=self.keep_top,
            optimize_batch_size=self.optimize_batch_size,
            optimize_n_epochs=self.optimize_n_epochs,
            pretrain_n_epochs=self.pretrain_n_epochs,
            database_name=database_name,
        )

        # take the molecules seen during the hill-climbing, and also sample from the final model
        samples = [m.smiles for m in molecules]
        if self.sample_final_model_only:
            samples.clear()
        print("Going to sample last batch of molecules")
        last_sample = generator.sample(max(number_molecules, self.number_final_samples))
        samples += last_sample

        raws_to_save = [
            {
                "benchmark": benchmark_name,
                "epoch": "Last",
                "smiles": smiles,
                "date": str(date.today()),
            }
            for smiles in last_sample
        ]
        self.client[database_name].insert_many(raws_to_save)
        if "pi3kmtor" in database_name:
            add_score_to_db_pi3kmtor(database_name)
        # calculate the scores and return the best ones
        samples = canonicalize_list(samples)
        scores = scoring_function.score_list(samples)

        scored_molecules = zip(samples, scores)
        sorted_scored_molecules = sorted(
            scored_molecules, key=lambda x: (x[1], hash(x[0])), reverse=True
        )

        top_scored_molecules = sorted_scored_molecules[:number_molecules]

        return [x[0] for x in top_scored_molecules]
