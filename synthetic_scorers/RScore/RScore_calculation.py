"""
Copyright (C) Iktos - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

import time
import requests
import os
from collections import Counter


def interact_with_api(smiles_list, timeout=1):
    url = os.getenv("SPAYA_API_URL")
    token = os.getenv("SPAYA_API_TOKEN")

    get_res = requests.post(
        url=url,
        json={
            "batch_smiles": smiles_list,
            "max_depth": 15,
            "nb_steps": 800,
            "early_stopping_score": 0.6,
            "early_stopping_timeout": timeout,
        },
        headers={"Authorization": "Bearer " + token},
    )
    return get_res.json()["smiles"]


def get_RScore(smiles_list) -> float:
    t1 = time.perf_counter()
    max_time = 70 * 60
    n_smiles = len(smiles_list)
    print("{} molecules to retrosynthesis trough Spaya API".format(n_smiles))
    i = 0
    smiles_list_0 = smiles_list
    results_all = {}
    try:
        while len(smiles_list) > 0 and time.perf_counter() - t1 < max_time:
            results = interact_with_api(smiles_list)
            i += 1
            results = {
                doc["smiles"]: {
                    "status": doc["status"],
                    "rscore": 0 if doc["rscore"] is None else doc["rscore"],
                }
                for doc in results
            }
            results_all.update(results)

            if i % 1 == 0:
                print(Counter([doc["status"] for key, doc in results_all.items()]))
            smiles_list = [
                s
                for s, val in results.items()
                if val["status"] not in ["DONE", "INVALID SMILES"]
            ]
            if len(smiles_list) > 0:
                if len(smiles_list) < 50:
                    time.sleep(10)
                else:
                    time_to_wait = (int(len(smiles_list) / 150) + 1) * 60
                    time.sleep(time_to_wait)

        if len(smiles_list) > 0:
            results = interact_with_api(smiles_list)
            results = {
                doc["smiles"]: {
                    "status": doc["status"],
                    "rscore": 0 if doc["rscore"] is None else doc["rscore"],
                }
                for doc in results
            }
            results_all.update(results)
            smiles_list = [
                s
                for s, val in results.items()
                if val["status"] not in ["DONE", "INVALID SMILES"]
            ]

    except KeyboardInterrupt:
        print("key board interruption")
    scores_0 = [results_all.get(s) for s in smiles_list_0]
    scores = [0 if doc is None else doc["rscore"] for doc in scores_0]
    print("score average : ", sum(scores) / len(scores))
    print("n smiles not done : ", len(smiles_list))
    print(
        "time to score  {} molecules in min : {}".format(
            n_smiles, int((time.perf_counter() - t1) / 60)
        )
    )

    return scores
