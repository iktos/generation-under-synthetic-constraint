"""
Copyright (C) Iktos - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""

#
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import math
from collections import defaultdict
from synthetic_scorers.sascore import fpscores_filename

_fscores = None


def readFragmentScores():
    import pickle, gzip

    global _fscores
    _fscores = pickle.load(gzip.open(fpscores_filename))
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    if ri is None:
        ri = mol.GetRingInfo()
    arings = [set(x) for x in ri.AtomRings()]
    spiros = set()
    for i, ari in enumerate(arings):
        for j in range(i + 1, len(arings)):
            shared = ari & arings[j]
            if len(shared) == 1:
                spiros.update(shared)
    nSpiro = len(spiros)

    # find bonds that are shared between rings that share at least 2 bonds:
    nBridge = 0
    brings = [set(x) for x in ri.BondRings()]
    bridges = set()
    for i, bri in enumerate(brings):
        for j in range(i + 1, len(brings)):
            shared = bri & brings[j]
            if len(shared) > 1:
                atomCounts = defaultdict(int)
                for bi in shared:
                    bond = mol.GetBondWithIdx(bi)
                    atomCounts[bond.GetBeginAtomIdx()] += 1
                    atomCounts[bond.GetEndAtomIdx()] += 1
                tmp = 0
                for ai, cnt in atomCounts.items():
                    if cnt == 1:
                        tmp += 1
                        bridges.add(ai)
                # if tmp!=2: # no need to stress the users
                # print 'huh:',tmp
    return len(bridges), nSpiro


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(
        m, 2
    )  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = (
        0.0
        - sizePenalty
        - stereoPenalty
        - spiroPenalty
        - bridgePenalty
        - macrocyclePenalty
    )

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
    # smooth the 10-end
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore
