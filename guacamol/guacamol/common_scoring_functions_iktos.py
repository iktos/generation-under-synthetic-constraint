"""
Copyright (C) Iktos - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

from typing import Callable
from rdkit import Chem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
import os
from guacamol.guacamol.utils.fingerprints import get_fingerprint
from guacamol.guacamol.score_modifier import ScoreModifier, MinGaussianModifier, MaxGaussianModifier
from guacamol.guacamol.scoring_function import ScoringFunctionBasedOnRdkitMol, MoleculewiseScoringFunction, BatchScoringFunction
from guacamol.guacamol.utils.chemistry import smiles_to_rdkit_mol
import time
import json
import numpy as np
import rdkit.Chem.QED
from synthetic_scorers.sascore.sascorer import calculateScore
from synthetic_scorers.scscore.standalone_model_numpy import SCScorer
from synthetic_scorers.RAscore.RAscore_NN import RAScorerNN
from guacamol.guacamol.utils.utils_iktos import compute_morgan_fp, add_descriptors
from synthetic_scorers.RScore.RScore_calculation import get_RScore
from synthetic_scorers.RSPred.predictorRS import RSPredictor

class RdkitScoringFunction(ScoringFunctionBasedOnRdkitMol):
    """
    Scoring function wrapping RDKit descriptors.
    """

    def __init__(self, descriptor: Callable[[Chem.Mol], float], score_modifier: ScoreModifier = None) -> None:
        """
        Args:
            descriptor: molecular descriptors, such as the ones in descriptors.py
            score_modifier: score modifier
        """
        super().__init__(score_modifier=score_modifier)
        self.descriptor = descriptor

    def score_mol(self, mol: Chem.Mol) -> float:
        return self.descriptor(mol)


class MaxTanimotoScoringFunction(ScoringFunctionBasedOnRdkitMol):
    """
    Scoring function that looks at the fingerprint similarity against a target molecule.
    """

    def __init__(self, smiles_ref, fp_type, score_modifier: ScoreModifier = None, ) -> None:
        """
        Args:
            targets: targets molecules
            fp_type: fingerprint type
            score_modifier: score modifier
        """
        super().__init__(score_modifier=score_modifier)

        self.fp_type = fp_type
        self.smiles_ref = smiles_ref
        targets_mols = [smiles_to_rdkit_mol(target) for target in smiles_ref]
        if None in targets_mols:
            raise RuntimeError(
                f'A similarity target is not a valid molecule.')

        self.ref_fp = [get_fingerprint(mol, self.fp_type) for mol in targets_mols]

    def score_mol(self, mol: Chem.Mol, return_smiles=False) -> float:
        fp = get_fingerprint(mol, self.fp_type)
        all_tani = np.array([TanimotoSimilarity(fp, ref ) for ref in self.ref_fp])
        if return_smiles :
            return self.smiles_ref[np.argmax(all_tani)]
        else:
            return np.max(all_tani)
        

    


url = "https://dev.retrosynthesis-api.maheo.tech/api/batch_smiles"
def isNaN(num):
    return num != num

class RScoreScoringFunction(BatchScoringFunction):
    def __init__(self, mu=0.7, sigma=0.2):
        """
        Args:
            target: target molecule
            score_modifier: score modifier
        """
        super().__init__()
        self.modifier = MaxGaussianModifier(mu, sigma)

    def score_list(self, list_smiles):
        max_batch = 2000
        try:
            if len(list_smiles) > max_batch:
                print("too many smiles - chunk ---")
                scores = []
                for i in range(0, len(list_smiles), max_batch):
                    scores_tmp = get_RScore(list_smiles[i:i + max_batch])
                    scores.extend(scores_tmp)

            else:
                scores = get_RScore(list_smiles)

            scores = [0 if v is None else self.modifier(v) for v in scores]
            return scores
    
        except:
            print(" -----  reconnecting   -----")
            time.sleep(5)
            scores = get_RScore(list_smiles)

            scores = [0 if v is None else self.modifier(v) for v in scores]
            return scores





class QEDScoringFunction(ScoringFunctionBasedOnRdkitMol):
    def __init__(self):
        super().__init__()
        mu = 0.6
        sigma = 0.13
        self.modifier = MaxGaussianModifier(mu, sigma)

    def score_mol(self, mol: Chem.Mol) -> float:
        score = rdkit.Chem.QED.qed(mol)
        score = self.modifier(score) if self.modifier else score
        return score


class pi3kScoringFunction(ScoringFunctionBasedOnRdkitMol):
    def __init__(self):
        super().__init__()
        weights_mtor_file = "pi3kmtor/weights_pi3k.json"
        with open(weights_mtor_file, "r") as f:
            weights_pi3k = json.load(f)
        b = weights_pi3k["intercept_"]
        a = weights_pi3k["coef_"]
        self.predictor = lambda x: np.dot(a, x) + b
        self.modifier = MaxGaussianModifier(8, 1)  # before (7,1)
        self.radius = 6
        self.nb_bits = 4096

    def score_mol(self, mol):
        morgan_fp = compute_morgan_fp(mol, self.nb_bits, self.radius)
        descriptors = add_descriptors(mol)
        features = np.concatenate((morgan_fp, descriptors), axis=0)
        pi3k_pred = self.predictor(features)
        pi3k_pred = self.modifier(pi3k_pred) if self.modifier else pi3k_pred

        return pi3k_pred


class mtorScoringFunction(ScoringFunctionBasedOnRdkitMol):
    def __init__(self):
        super().__init__()
        weights_mtor_file = "pi3kmtor/weights_mtor.json"
        with open(weights_mtor_file, "r") as f:
            weights_mtor = json.load(f)
        b = weights_mtor["intercept_"]
        a = weights_mtor["coef_"]
        self.predictor = lambda x: np.dot(a, x) + b
        self.modifier = MaxGaussianModifier(9.5, 1)  # before(8.5,1)
        self.radius = 4
        self.nb_bits = 4096

    def score_mol(self, mol):
        morgan_fp = compute_morgan_fp(mol, self.nb_bits, self.radius)
        mtor_pred = self.predictor(morgan_fp)
        mtor_pred = self.modifier(mtor_pred) if self.modifier else mtor_pred

        return mtor_pred


class SAScoringFunction(ScoringFunctionBasedOnRdkitMol):

    def __init__(self, mu=2.5, sigma=0.4):
        super().__init__()
        self.modifier = MinGaussianModifier(mu, sigma) # target 2.8

    def score_mol(self, mol: Chem.Mol) -> float:
        return self.modifier(calculateScore(mol))


class SCScoringFunction(MoleculewiseScoringFunction):
    def __init__(self, mu=4.3, sigma=0.2):
        super().__init__()

        self.scscorer = SCScorer()
        self.modifier = MinGaussianModifier(mu, sigma) # target 4.5

    def raw_score(self, smiles: str) -> float:
        return self.modifier(self.scscorer.get_score_from_smi(smiles)[1])
    


class RAScoringFunction(MoleculewiseScoringFunction):
    def __init__(self, mu=0.7, sigma=0.2):
        super().__init__()
        
        path_model = "synthetic_scorers/RAscore/models/DNN_chembl_fcfp_counts/model.h5"
        self.RAscorer = RAScorerNN(model_path=path_model)
        self.modifier = MaxGaussianModifier(mu, sigma) 

    def raw_score(self, smiles: str) -> float:
        return self.modifier(self.RAscorer.predict(smiles))
    
    

class ImposeStructure(ScoringFunctionBasedOnRdkitMol):
    def __init__(self, smarts_structure):
        super().__init__()
# essai

        self.structure_mol = Chem.MolFromSmarts(smarts_structure)
   

    def score_mol(self, mol: Chem.Mol) -> float:
        try:
            is_good = mol.HasSubstructMatch(self.structure_mol)
        except RuntimeError:
            is_good = 0
        return is_good
                

    
    
class RSPredScoringFunction(BatchScoringFunction):
    def __init__(self, weights_filename=None, mu =0.7, sigma=0.2):
        self.model= RSPredictor(weights_filename)
        self.modifier = MaxGaussianModifier(mu, sigma)

    def score_list(self, list_smiles):
        if len(list_smiles)==0:
            return []
        list_scores = self.model.predict_many(list_smiles)
        return [self.modifier(x) for x in list_scores]
    
    
    