"""
Copyright (C) Iktos - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
"""


import pickle
from typing import List
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import Tensor
from torch import nn
import os

class RScorePredictorNet(nn.Module):  # predict continous score in [0, 1]

    def __init__(self, input_size, hidden_size, dropout_p):
        super(RScorePredictorNet, self).__init__()
        self.first_layer = nn.Linear(input_size, hidden_size)
        self.second_layer = nn.Linear(hidden_size, hidden_size)
        self.third_layer = nn.Linear(hidden_size, hidden_size)
        self.final_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.elu_act = nn.ReLU()
        self.sigmoid_act = nn.Sigmoid()
        self.batchnorm_1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm_2 = nn.BatchNorm1d(hidden_size)
        self.batchnorm_3 = nn.BatchNorm1d(hidden_size)

        
    def forward(self, x):
        x = self.first_layer(x)
        x = self.batchnorm_1(x)
        x = self.elu_act(x)
        x = self.dropout(x)
        x = self.second_layer(x)
        x = self.batchnorm_2(x)
        x = self.elu_act(x)
        x = self.dropout(x)
        x = self.third_layer(x)
        x = self.batchnorm_3(x)
        x = self.elu_act(x)
        x = self.dropout(x)
        x = self.final_layer(x)
        x = self.sigmoid_act(x)
        return x


def compute_morgan_fp(mol, radius, n_bits):
    fp = np.zeros(n_bits, dtype=int)#

    morgan_fp = AllChem.GetHashedMorganFingerprint(
        mol, radius=radius, nBits=n_bits, useChirality=True
    ).GetNonzeroElements()

    np.put(fp, list(morgan_fp.keys()), list(morgan_fp.values()))

    return fp



            
class RSPredictor:
    def __init__(self, weights_filename=None):
            if weights_filename is None:
                weights_filename = "chembl_230K_final_87.0.pickle"

            weights_path = os.path.join(os.path.dirname(__file__), "models", weights_filename)
            state_dict = pickle.load(open(weights_path, 'rb'))
            params = state_dict["params"]
            self.radius = params["radius"]
            self.morgan_size = params["morgan_size"]
            self.my_predictor = RScorePredictorNet(input_size=params["morgan_size"], hidden_size=params["hidden_size"], dropout_p=params["dropout_p"])
            self.my_predictor.load_state_dict(state_dict['state_dict'])
            self.my_predictor.eval()

    def compute_fingerprint(self, mol):
        fp = compute_morgan_fp(mol, radius=self.radius, n_bits=self.morgan_size)
        return fp

    def predict(self, smiles):
        fp = self.compute_fingerprint(Chem.MolFromSmiles(smiles))
        fp = torch.log1p(Tensor(fp)).view(1,-1)
        with torch.no_grad():
            proba_synth = self.my_predictor(fp).item()
            return proba_synth

    def predict_many(self, list_smiles):
        mol_list = [Chem.MolFromSmiles(s) for s in list_smiles]
        id_nul = [i for i in range(len(mol_list)) if mol_list[i] is None]
        mol_list = [m for m in mol_list if m is not None]
        fp_list = [self.compute_fingerprint(mol) for mol in mol_list]
        try:
            fp_list = torch.log1p(Tensor(fp_list)).view(len(mol_list),-1)
        except:
            print("PBM BECAUSE LEN MOL LIST IS : ", len(mol_list))
        with torch.no_grad():
            proba_synth = list(self.my_predictor(fp_list).view(-1).numpy())
            for id_ in id_nul:
                proba_synth.insert(id_, 0)
            return proba_synth

