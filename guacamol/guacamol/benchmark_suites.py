from typing import List, Optional

from guacamol.guacamol.goal_directed_benchmark import GoalDirectedBenchmark

from guacamol.guacamol.standard_benchmarks_iktos import pi3kmtor_bench, logP_benchmark_iktos, \
    cns_mpo_iktos, tpsa_benchmark_iktos, qed_benchmark_iktos, hard_osimertinib_and_synth,\
    hard_fexofenadine_and_synth, ranolazine_mpo_and_synth, perindopril_rings_and_synth,\
    amlodipine_rings_and_synth, sitagliptin_replacement_and_synth, zaleplon_with_other_formula_and_synth,\
    valsartan_smarts_and_synth, decoration_hop_and_synth, scaffold_hop_and_synth, isomers_c7h8n2o2_and_synth, similarity_and_synth


def goal_directed_benchmark_suite(version_name: str, synth_score: Optional[str]) -> List[GoalDirectedBenchmark]:
    print("version name : ", version_name)
    if version_name == "guacamol_paper":
        return guacamol_generations_paper()

    if version_name == "pi3kmtor_paper":
        return pi3kmtor_generations_paper()

    if version_name == "chembl":
        return goal_directed_suite_chembl_simple()

    if version_name == "pi3kmtor":
        return [pi3kmtor_bench(synth_score)]

    if version_name=="trivial":
        return goal_directed_suite_trivial(synth_score)

    if version_name=="test":
        return goal_directed_suite_test()

    raise Exception(f'Goal-directed benchmark suite "{version_name}" does not exist.')


def guacamol_generations_paper() -> List[GoalDirectedBenchmark]:
    """
    Launches all the generations on guacamol benchmarks used.
    - Classic generations
    - Generations with SAscore constraint
    - Generaion with RScore constraint
    """
    all_generations = []
    for score_synth in [None, "SAscore", "RScore", "RSPred"]:
        generations = [#hard_osimertinib_and_synth(score_synth=score_synth),
        hard_fexofenadine_and_synth(score_synth=score_synth),
        ranolazine_mpo_and_synth(score_synth=score_synth),
        perindopril_rings_and_synth(score_synth=score_synth),
        amlodipine_rings_and_synth(score_synth=score_synth),
        sitagliptin_replacement_and_synth(score_synth=score_synth),
        zaleplon_with_other_formula_and_synth(score_synth=score_synth),
        valsartan_smarts_and_synth(score_synth=score_synth),
        decoration_hop_and_synth(score_synth=score_synth),
        scaffold_hop_and_synth(score_synth=score_synth)]
        all_generations.extend(generations)
    return all_generations

def pi3kmtor_generations_paper() -> List[GoalDirectedBenchmark]:
    """
    Launches all the generations of pi3kmtor.
    One classic generation, and 5 generations under a synthetic score constraint.
    """
    all_generations = []
    list_score_synth = [None, "SAscore", "RAscore","SCscore", "RSPred", "RScore"]
    for score_synth in list_score_synth:
        all_generations.append(pi3kmtor_bench(score_synth))
    return all_generations



def goal_directed_suite_chembl_simple() -> List[GoalDirectedBenchmark]:
    """
    Launches the trivial tasks of guacamol benchmark, in the classic case and with
    the RScore constraint.
    """
    all_generations = []
    for score_synth in [None, "RScore"]:
        generations = [
            logP_benchmark_iktos(target=-1.0, score_synth=score_synth),
            cns_mpo_iktos(with_RScore=False, score_synth=score_synth),
            tpsa_benchmark_iktos(target=150.0, score_synth=score_synth),
            qed_benchmark_iktos(score_synth=score_synth)]
        all_generations.extend(generations)
    return all_generations


def goal_directed_suite_trivial(synth_score) -> List[GoalDirectedBenchmark]:
    """
    Trivial goal-directed benchmarks from the paper.
    """
    return [
        logP_benchmark_iktos(target=-1.0, score_synth=synth_score ),
        logP_benchmark_iktos(target=8.0, score_synth=synth_score),
        tpsa_benchmark_iktos(target=150.0, score_synth=synth_score),
        cns_mpo_iktos(score_synth=synth_score),
        qed_benchmark_iktos(score_synth=synth_score),
        isomers_c7h8n2o2_and_synth(score_synth=synth_score),
    ]



def goal_directed_suite_test() -> List[GoalDirectedBenchmark]:
    """
    Trivial goal-directed benchmarks from the paper.
    """
    return [
        logP_benchmark_iktos(target=-1.0),
        pi3kmtor_bench(None),
        pi3kmtor_bench("SAscore"),
        pi3kmtor_bench("SCscore"),
        pi3kmtor_bench("RAscore"),
        pi3kmtor_bench("RSPred"),
        ranolazine_mpo_and_synth(None),
    ]
