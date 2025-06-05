from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AllChem import GetMorganGenerator, GetRDKitFPGenerator
from .pubchem_fp import GetPubChemFPs
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# 圆形分子指纹
def morgan_binary_features_generator(mol, radius: int = 2,
                                     num_bits: int = 2048):
    mol = Chem.MolFromSmiles(mol) if type(mol) is str else mol
    fpgen = GetMorganGenerator(radius=radius, fpSize=num_bits)
    features_vec = fpgen.GetFingerprint(mol)
    features = np.array(features_vec)
    return features

#
def rdk_features_generator(mol):
    mol = Chem.MolFromSmiles(mol) if type(mol) is str else mol
    fpgen = GetRDKitFPGenerator()
    features_vec = fpgen.GetFingerprint(mol)
    features = np.array(features_vec)
    return features


# ** 基于字典
def maccs_features_generator(mol):
    mol = Chem.MolFromSmiles(mol) if type(mol) is str else mol
    features_vec = MACCSkeys.GenMACCSKeys(mol)
    features = np.array(features_vec)
    return features

# ***
def pubchem_features_generator(mol):
    features = GetPubChemFPs(mol)
    return features



if __name__ == '__main__':
    smi = 'O=C1CCCO1'
    fp = GetPubChemFPs(smi)
    print(fp)
