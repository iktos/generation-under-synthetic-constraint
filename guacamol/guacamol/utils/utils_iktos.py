"""
Copyright (C) Iktos - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

from rdkit.Chem import Descriptors
from sklearn.linear_model import LinearRegression
import json
from rdkit.Chem import Descriptors
import numpy as np
from rdkit.Chem import AllChem


def compute_morgan_fp(mol, nb_bits, radius) -> np.array:
    fp = np.zeros(nb_bits, dtype=int)
    morgan_fp = AllChem.GetHashedMorganFingerprint(
        mol, radius=radius, nBits=nb_bits, useChirality=False, useBondTypes=False, useFeatures=False
    ).GetNonzeroElements()
    fp_values = np.array((list(morgan_fp.values())))>0
    np.put(fp, list(morgan_fp.keys()), fp_values)
    return fp

def add_descriptors(mol) :  # pragma: no cover
    """Computes 2D Descriptors for a given SMILES
    """
 #   mol = MolFromSmiles(smiles)
    descriptors = [
        Descriptors.TPSA(mol),
        Descriptors.SlogP_VSA1(mol),
        Descriptors.SlogP_VSA2(mol),
        Descriptors.SlogP_VSA3(mol),
        Descriptors.SlogP_VSA4(mol),
        Descriptors.SlogP_VSA5(mol),
        Descriptors.SlogP_VSA6(mol),
        Descriptors.SlogP_VSA7(mol),
        Descriptors.SlogP_VSA8(mol),
        Descriptors.SlogP_VSA9(mol),
        Descriptors.SlogP_VSA10(mol),
        Descriptors.SlogP_VSA11(mol),
        Descriptors.SlogP_VSA12(mol),
        Descriptors.MolLogP(mol),
        Descriptors.MolMR(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumAromaticHeterocycles(mol),
        Descriptors.NumSaturatedHeterocycles(mol),
        Descriptors.NumAliphaticHeterocycles(mol),
        Descriptors.NumAromaticCarbocycles(mol),
        Descriptors.NumSaturatedCarbocycles(mol),
        Descriptors.NumAliphaticCarbocycles(mol),
    ]
    return np.asarray(descriptors)