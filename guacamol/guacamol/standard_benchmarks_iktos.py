"""
Copyright (C) Iktos - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

from rdkit import Chem

from guacamol.guacamol.common_scoring_functions import TanimotoScoringFunction, RdkitScoringFunction, CNS_MPO_ScoringFunction, \
    IsomerScoringFunction, SMARTSScoringFunction,   MaxTanimotoScoringFunction
from guacamol.guacamol.common_scoring_functions_iktos import  RScoreScoringFunction, MaxTanimotoScoringFunction, QEDScoringFunction, \
    mtorScoringFunction,pi3kScoringFunction, SAScoringFunction, SCScoringFunction, RAScoringFunction, ImposeStructure, RSPredScoringFunction

from guacamol.guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.guacamol.goal_directed_score_contributions import uniform_specification
from guacamol.guacamol.score_modifier import MinGaussianModifier, MaxGaussianModifier, ClippedScoreModifier, GaussianModifier
from guacamol.guacamol.scoring_function import ArithmeticMeanScoringFunction, GeometricMeanScoringFunction, ScoringFunction, BatchGeometricMeanScoringFunction
from guacamol.guacamol.utils.descriptors import num_aromatic_rings, logP, qed, tpsa, bertz, \
    AtomCounter, num_rings

import os

def add_synth_scorer(list_scoring, name, mean_cls, score_synth,path_weights=None):
    if score_synth is not None:
        if score_synth == "RScore":
            scoreSynth = RScoreScoringFunction()
            os.environ["SPAYA_API_URL"] = "https://dev.retrosynthesis-api.maheo.tech/api/batch_smiles"
            os.environ["SPAYA_API_TOKEN"] = "KYifdgUw&nbBlGBqhosLvq0YnL"

            if not os.getenv("SPAYA_API_TOKEN"):
                raise Exception("No token for SpayaAPI was set in environement "
                                "variable SPAYA_API_TOKEN")
            if not os.getenv("SPAYA_API_URL"):
                raise Exception(
                    "No  SPAYA_API_URL was set in environement variable")

            mean_cls = BatchGeometricMeanScoringFunction
        elif score_synth == "SAscore":
            scoreSynth = SAScoringFunction()
        elif score_synth == "SCscore":
            scoreSynth = SCScoringFunction()
        elif score_synth == "RAscore":
            scoreSynth = RAScoringFunction()
        elif score_synth == "RSPred":
            weights_filename = "chembl_230K_final_87.0.pickle"
            scoreSynth = RSPredScoringFunction(weights_filename=weights_filename)
        list_scoring.append(scoreSynth)
        name = name+score_synth

    scoring = mean_cls(list_scoring)
    return scoring, name


def pi3kmtor_bench(score_synth) -> GoalDirectedBenchmark:
    mean_cls = GeometricMeanScoringFunction
    name = "pi3kmtor"
    with open("pi3kmtor/pi3kmtor.smiles", "r") as f:
        smiles_ref = f.readlines()
    similarity = MaxTanimotoScoringFunction(smiles_ref=smiles_ref, fp_type="ECFP4",  score_modifier=MaxGaussianModifier(mu=0.75, sigma=0.25)) # before (0.5, 0.1)
    QED = QEDScoringFunction()
    pi3k = pi3kScoringFunction()
    mtor = mtorScoringFunction()
    list_scoring = [similarity, QED, pi3k, mtor]
    filter_structure = ImposeStructure("c1cncc(c1)C#Cc1cncnc1")
    list_scoring.append(filter_structure)
    scoring, name = add_synth_scorer(list_scoring, name, mean_cls, score_synth)
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(name=name,
                                 objective=scoring,
                                 objective_start=similarity,
                                 contribution_specification=specification)


def similarity_and_synth(smiles: str, name: str, score_synth, fp_type: str = 'ECFP4',threshold: float = 0.7,
               rediscovery: bool = False, mean_cls = GeometricMeanScoringFunction, ) -> GoalDirectedBenchmark:
    category = 'rediscovery' if rediscovery else 'similarity'
    benchmark_name = f'{name} {category}'

    modifier = ClippedScoreModifier(upper_x=threshold)
    tanimoto = TanimotoScoringFunction(target=smiles, fp_type=fp_type, score_modifier=modifier)
    list_scoring = [tanimoto]
    if rediscovery:
        specification = uniform_specification(1)
    else:
        specification = uniform_specification(1, 10, 100)
    scoring_function, name = add_synth_scorer(list_scoring, name, mean_cls,score_synth)
    return GoalDirectedBenchmark(name=benchmark_name,
                                 objective=scoring_function,
                                 objective_start=tanimoto,
                                 contribution_specification=specification)

def median_camphor_menthol_and_synth( score_synth, mean_cls=GeometricMeanScoringFunction,) -> GoalDirectedBenchmark:
    s1 = 'CC1(C)C2CCC1(C)C(=O)C2'
    s2 = 'CC(C)C1CCC(C)CC1O'
    name = "Median molecules 1"

    t_camphor = TanimotoScoringFunction(s1, fp_type='ECFP4')
    t_menthol = TanimotoScoringFunction(s2, fp_type='ECFP4')
    list_scoring = [t_camphor, t_menthol]
    scoring_function_start = mean_cls( [t_camphor, t_menthol])
    scoring_function, name = add_synth_scorer(list_scoring, name,mean_cls, score_synth)
    specification = uniform_specification(1, 10, 100)
    return GoalDirectedBenchmark(name=name,
                                 objective=scoring_function,
                                objective_start=scoring_function_start,
                                 contribution_specification=specification)


def median_tadalafil_sildenafil_and_synth(score_synth)-> GoalDirectedBenchmark:
    # median mol between tadalafil and sildenafil
    s1 = 'O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C'
    s2 = 'CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C'
    m1 = TanimotoScoringFunction(s1, fp_type='ECFP6')
    m2 = TanimotoScoringFunction(s2, fp_type='ECFP6')
    list_scoring = [m1, m2]
    mean_cls = GeometricMeanScoringFunction
    name = 'Median molecules 2'
    scoring_function_start = mean_cls([m1, m2])
    scoring_function, name = add_synth_scorer(list_scoring, name, mean_cls, score_synth)
    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=name,
                                 objective=scoring_function,
                                objective_start=scoring_function_start,
                                 contribution_specification=specification)


def hard_osimertinib_and_synth(score_synth, mean_cls=GeometricMeanScoringFunction) -> GoalDirectedBenchmark:
    smiles = 'COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34'
    name = 'Osimertinib MPO'
    modifier = ClippedScoreModifier(upper_x=0.8)
    similar_to_osimertinib = TanimotoScoringFunction(smiles, fp_type='FCFP4', score_modifier=modifier)

    but_not_too_similar = TanimotoScoringFunction(smiles, fp_type='ECFP6',
                                                  score_modifier=MinGaussianModifier(mu=0.85, sigma=0.1))

    tpsa_over_100 = RdkitScoringFunction(descriptor=tpsa,
                                         score_modifier=MaxGaussianModifier(mu=100, sigma=10))

    logP_scoring = RdkitScoringFunction(descriptor=logP,
                                        score_modifier=MinGaussianModifier(mu=1, sigma=1))
    list_scoring = [similar_to_osimertinib,but_not_too_similar,tpsa_over_100 , logP_scoring]
    scoring_function_start = mean_cls(list_scoring)
    make_osimertinib_great_again, name = add_synth_scorer(list_scoring, name, mean_cls, score_synth)
    
    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=name,
                                 objective=make_osimertinib_great_again,
                                objective_start=scoring_function_start,
                                 contribution_specification=specification)


def hard_fexofenadine_and_synth(score_synth, mean_cls=GeometricMeanScoringFunction) -> GoalDirectedBenchmark:
    """
    make fexofenadine less greasy
    :return:
    """
    smiles = 'CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4'
    name = 'Fexofenadine MPO'
    modifier = ClippedScoreModifier(upper_x=0.8)
    similar_to_fexofenadine = TanimotoScoringFunction(smiles, fp_type='AP', score_modifier=modifier)

    tpsa_over_90 = RdkitScoringFunction(descriptor=tpsa,
                                        score_modifier=MaxGaussianModifier(mu=90, sigma=10))

    logP_under_4 = RdkitScoringFunction(descriptor=logP,
                                        score_modifier=MinGaussianModifier(mu=4, sigma=1))
    
    list_scoring = [similar_to_fexofenadine, tpsa_over_90, logP_under_4] 

    scoring_function_start = mean_cls(list_scoring)

    optimize_fexofenadine, name = add_synth_scorer(list_scoring, name, mean_cls, score_synth)
    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=name,
                                 objective=optimize_fexofenadine,
                                 objective_start=scoring_function_start,
                                 contribution_specification=specification)


def ranolazine_mpo_and_synth(score_synth, mean_cls=GeometricMeanScoringFunction,) -> GoalDirectedBenchmark:
    """
    Make start_pop_ranolazine more polar and add a fluorine
    """
    ranolazine = 'COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2'
    name = 'Ranolazine MPO'
    modifier = ClippedScoreModifier(upper_x=0.7)
    similar_to_ranolazine = TanimotoScoringFunction(ranolazine, fp_type='AP', score_modifier=modifier)

    logP_under_4 = RdkitScoringFunction(descriptor=logP, score_modifier=MaxGaussianModifier(mu=7, sigma=1))

    tpsa_f = RdkitScoringFunction(descriptor=tpsa, score_modifier=MaxGaussianModifier(mu=95, sigma=20))

    fluorine = RdkitScoringFunction(descriptor=AtomCounter('F'), score_modifier=GaussianModifier(mu=1, sigma=1.0))

    list_scoring = [similar_to_ranolazine, logP_under_4, tpsa_f,fluorine]

    optimize_ranolazine, name = add_synth_scorer(list_scoring, name, mean_cls, score_synth)
                                                
    scoring_function_start = GeometricMeanScoringFunction( [similar_to_ranolazine, logP_under_4, tpsa_f,fluorine] )

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=name,
                                 objective=optimize_ranolazine,
                                objective_start=scoring_function_start,
                                 contribution_specification=specification,
                                 starting_population=[ranolazine])



def perindopril_rings_and_synth(score_synth, ) -> GoalDirectedBenchmark:
    # perindopril with two aromatic rings
    smiles = 'O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC'
    name = 'Perindopril MPO'
    perindopril = TanimotoScoringFunction(smiles,
                                          fp_type='ECFP4')
    arom_rings = RdkitScoringFunction(descriptor=num_aromatic_rings,
                                      score_modifier=GaussianModifier(mu=2, sigma=0.5))
    list_scoring = [perindopril, arom_rings]
    mean_cls = GeometricMeanScoringFunction
 
    specification = uniform_specification(1, 10, 100)
    
    scoring_function_start = mean_cls([perindopril, arom_rings])
    scoring_function, name = add_synth_scorer(list_scoring, name, mean_cls, score_synth)
    return GoalDirectedBenchmark(name=name,
                                 objective=scoring_function,
                                 objective_start = scoring_function_start,
                                 contribution_specification=specification)


def perindopril_rings_and_synth(score_synth) -> GoalDirectedBenchmark:
    # perindopril with two aromatic rings
    mean_cls = GeometricMeanScoringFunction
    smiles = 'O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC'
    name = 'Perindopril MPO'
    perindopril = TanimotoScoringFunction(smiles,
                                          fp_type='ECFP4')
    arom_rings = RdkitScoringFunction(descriptor=num_aromatic_rings,
                                      score_modifier=GaussianModifier(mu=2, sigma=0.5))

    list_scoring = [perindopril, arom_rings]

    specification = uniform_specification(1, 10, 100)

    scoring_function_start = GeometricMeanScoringFunction([perindopril, arom_rings])
    scoring_function, name = add_synth_scorer(list_scoring, name, mean_cls, score_synth)
    return GoalDirectedBenchmark(name=name,
                                 objective=scoring_function,
                                 objective_start=scoring_function_start,
                                 contribution_specification=specification)


def amlodipine_rings_and_synth(score_synth) -> GoalDirectedBenchmark:
    # amlodipine with 3 rings
    smiles = "Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC"
    name='Amlodipine MPO'
    amlodipine = TanimotoScoringFunction(smiles, fp_type='ECFP4')
    rings = RdkitScoringFunction(descriptor=num_rings,
                                 score_modifier=GaussianModifier(mu=3, sigma=0.5))
    list_scoring = [amlodipine, rings]
    mean_cls =GeometricMeanScoringFunction
  
    specification = uniform_specification(1, 10, 100)
    
    scoring_function_start = mean_cls([amlodipine, rings])
    scoring_function, name = add_synth_scorer(list_scoring, name, mean_cls, score_synth)

    return GoalDirectedBenchmark(name=name,
                                 objective=scoring_function,
                                objective_start = scoring_function_start,
                                 contribution_specification=specification)


def sitagliptin_replacement_and_synth(score_synth) -> GoalDirectedBenchmark:
    # Find a molecule dissimilar to sitagliptin, but with the same properties
    smiles = 'Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F'
    name = 'Sitagliptin MPO'
    sitagliptin = Chem.MolFromSmiles(smiles)
    target_logp = logP(sitagliptin)
    target_tpsa = tpsa(sitagliptin)

    mean_cls = GeometricMeanScoringFunction
    similarity = TanimotoScoringFunction(smiles, fp_type='ECFP4',
                                         score_modifier=GaussianModifier(mu=0, sigma=0.1))
    lp = RdkitScoringFunction(descriptor=logP,
                              score_modifier=GaussianModifier(mu=target_logp, sigma=0.2))
    tp = RdkitScoringFunction(descriptor=tpsa,
                              score_modifier=GaussianModifier(mu=target_tpsa, sigma=5))
    isomers = IsomerScoringFunction('C16H15F6N5O')
    list_scoring = [similarity, lp, tp, isomers]

    specification = uniform_specification(1, 10, 100)
    
    scoring_function_start = mean_cls([similarity, lp, tp, isomers])
    scoring_function, name = add_synth_scorer(list_scoring, name, mean_cls, score_synth)
    return GoalDirectedBenchmark(name=name,
                                 objective=scoring_function,
                                objective_start = scoring_function_start,
                                 contribution_specification=specification)


def zaleplon_with_other_formula_and_synth(score_synth, mean_cls=GeometricMeanScoringFunction,) -> GoalDirectedBenchmark:
    # zaleplon_with_other_formula with other formula
    smiles  = 'O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1'
    name='Zaleplon MPO'
    zaleplon = TanimotoScoringFunction(smiles,
                                       fp_type='ECFP4')
    formula = IsomerScoringFunction('C19H17N3O2')
    list_scoring = [zaleplon, formula]

    specification = uniform_specification(1, 10, 100)
    
    scoring_function_start = mean_cls([zaleplon, formula ])
    scoring_function, name = add_synth_scorer(list_scoring, name, mean_cls, score_synth)

    return GoalDirectedBenchmark(name=name,
                                 objective=scoring_function,
                                 objective_start = scoring_function_start,
                                 contribution_specification=specification)


def smarts_with_other_target_and_synth(smarts: str, other_molecule: str, score_synth) -> ScoringFunction:
    smarts_scoring_function = SMARTSScoringFunction(target=smarts)
    other_mol = Chem.MolFromSmiles(other_molecule)
    name = "Valsartan SMARTS"
    target_logp = logP(other_mol)
    target_tpsa = tpsa(other_mol)
    target_bertz = bertz(other_mol)

    lp = RdkitScoringFunction(descriptor=logP,
                              score_modifier=GaussianModifier(mu=target_logp, sigma=0.2))
    tp = RdkitScoringFunction(descriptor=tpsa,
                              score_modifier=GaussianModifier(mu=target_tpsa, sigma=5))
    bz = RdkitScoringFunction(descriptor=bertz,
                              score_modifier=GaussianModifier(mu=target_bertz, sigma=30))
    
    list_scoring = [smarts_scoring_function, lp, tp, bz]
    mean_cls = GeometricMeanScoringFunction

    scoring_function_start = mean_cls([smarts_scoring_function, lp, tp, bz])
    scoring_function, name = add_synth_scorer(list_scoring, name, mean_cls, score_synth)

    return scoring_function, scoring_function_start, name

def isomers_c7h8n2o2_and_synth(mean_function='geometric', score_synth=None) -> GoalDirectedBenchmark:
    """
    Benchmark to try and get 100 isomers for C7H8N2O2.

    Args:
        mean_function: 'arithmetic' or 'geometric'
    """

    specification = uniform_specification(100)

    objective_start = IsomerScoringFunction('C7H8N2O2', mean_function=mean_function)
    mean_cls = GeometricMeanScoringFunction

    objective, name = add_synth_scorer([objective_start], "isomers_c7h8n2o2", mean_cls=mean_cls, score_synth=score_synth)
    return GoalDirectedBenchmark(name=name,
                                 objective=objective,
                                 objective_start=objective_start,
                                 contribution_specification=specification)


def valsartan_smarts_and_synth(score_synth, mean_cls=GeometricMeanScoringFunction ) -> GoalDirectedBenchmark:
    # valsartan smarts with sitagliptin properties
    sitagliptin_smiles = 'NC(CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F'
    valsartan_smarts = 'CN(C=O)Cc1ccc(c2ccccc2)cc1'
   
    specification = uniform_specification(1, 10, 100)
    objective, scoring_function_start, name = smarts_with_other_target_and_synth(valsartan_smarts, sitagliptin_smiles, score_synth)

    return GoalDirectedBenchmark(name=name,
                                 objective=objective,
                                 objective_start = scoring_function_start,
                                 contribution_specification=specification)



def decoration_hop_and_synth(score_synth) -> GoalDirectedBenchmark:
    smiles = 'CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C'

    pharmacophor_sim = TanimotoScoringFunction(smiles, fp_type='PHCO',
                                               score_modifier=ClippedScoreModifier(upper_x=0.85))
    name = 'Deco Hop'
    # change deco
    deco1 = SMARTSScoringFunction('CS([#6])(=O)=O', inverse=True)
    deco2 = SMARTSScoringFunction('[#7]-c1ccc2ncsc2c1', inverse=True)
  # keep scaffold
    scaffold = SMARTSScoringFunction('[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12', inverse=False)

    list_scoring = [pharmacophor_sim, deco1, deco2, scaffold]

    mean_cls = ArithmeticMeanScoringFunction
    deco_hop1_fn = mean_cls(list_scoring)
    scoring_function_start = ArithmeticMeanScoringFunction([pharmacophor_sim, deco1, deco2, scaffold])
    scoring_function, name = add_synth_scorer(list_scoring, name, mean_cls, score_synth)
    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=name,
                                 objective=deco_hop1_fn,
                                objective_start = scoring_function_start,

                                 contribution_specification=specification)


def scaffold_hop_and_synth(score_synth, mean_cls=GeometricMeanScoringFunction,) -> GoalDirectedBenchmark:
    """
    Keep the decoration, and similarity to start point, but change the scaffold.
    """

    smiles = 'CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C'
    name = "Scaffold Hop"
    pharmacophor_sim = TanimotoScoringFunction(smiles, fp_type='PHCO',
                                               score_modifier=ClippedScoreModifier(upper_x=0.75))

    deco = SMARTSScoringFunction('[#6]-[#6]-[#6]-[#8]-[#6]~[#6]~[#6]~[#6]~[#6]-[#7]-c1ccc2ncsc2c1', inverse=False)

    # anti scaffold
    scaffold = SMARTSScoringFunction('[#7]-c1n[c;h1]nc2[c;h1]c(-[#8])[c;h0][c;h1]c12', inverse=True)
    
    list_scoring = [pharmacophor_sim, deco, scaffold]
    

  #  scaffold_hop_obj = ArithmeticMeanScoringFunction(list_scoring)
    scoring_function_start = ArithmeticMeanScoringFunction( [pharmacophor_sim, deco, scaffold])
    scoring_function, name = add_synth_scorer(list_scoring, name, mean_cls, score_synth)
    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=name,
                                 objective=scoring_function,
                                objective_start = scoring_function_start,

                                 contribution_specification=specification)

def chembl_only_spaya() -> GoalDirectedBenchmark:
    name = "chembl_bench_only_spaya"

    scoring = RScoreScoringFunction()

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=name,
                                 objective=scoring,
                                 objective_start=scoring,
                                 contribution_specification=specification)


def logP_benchmark_iktos(target: float, score_synth=None) -> GoalDirectedBenchmark:
    benchmark_name = f'logP (target: {target})'
    
    logp_scoring = RdkitScoringFunction(descriptor=logP,
                                     score_modifier=GaussianModifier(mu=target, sigma=1))
    scoring = logp_scoring
    scoring, benchmark_name = add_synth_scorer([scoring], benchmark_name, GeometricMeanScoringFunction, score_synth)
    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=benchmark_name,
                                 objective=scoring,
                                 objective_start = logp_scoring,
                                 contribution_specification=specification)


def tpsa_benchmark_iktos(target: float, score_synth) -> GoalDirectedBenchmark:
    benchmark_name = f'TPSA (target: {target})'
    tpsa_scoring = RdkitScoringFunction(descriptor=tpsa,
                                     score_modifier=GaussianModifier(mu=target, sigma=20.0))
    scoring=tpsa_scoring
    specification = uniform_specification(1, 10, 100)
    mean_cls = GeometricMeanScoringFunction
    scoring_function, benchmark_name = add_synth_scorer([scoring], benchmark_name, mean_cls, score_synth)
    return GoalDirectedBenchmark(name=benchmark_name,
                                 objective=scoring_function,
                                 objective_start=tpsa_scoring,
                                 contribution_specification=specification)


def cns_mpo_iktos(score_synth, max_logP=5.0) -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    name = "CNS MPO"
    cns_scoring = CNS_MPO_ScoringFunction(max_logP=max_logP)
    scoring = cns_scoring
    mean_cls = GeometricMeanScoringFunction
    scoring_function, benchmark_name = add_synth_scorer([scoring], name, mean_cls, score_synth)
    return GoalDirectedBenchmark(name=benchmark_name,
                                 objective=scoring_function,
                                 objective_start=cns_scoring,
                                 contribution_specification=specification)


def qed_benchmark_iktos(score_synth,) -> GoalDirectedBenchmark:
    specification = uniform_specification(1, 10, 100)
    name = "QED"
    qed_scoring = RdkitScoringFunction(descriptor=qed)
    scoring = qed_scoring
    mean_cls = GeometricMeanScoringFunction
    scoring, name = add_synth_scorer([scoring], name, mean_cls, score_synth)

    return GoalDirectedBenchmark(name=name,
                                 objective=scoring,
                                 objective_start=qed_scoring,  
                                 contribution_specification=specification)
