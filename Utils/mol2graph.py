import pandas as pd
import numpy as np
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer
from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader



def data_process(dataset,batch_size,label_str,shuffle=True):
  atom_featurizer = MultiHotAtomFeaturizer.v2()
  bond_featurizer = MultiHotBondFeaturizer()
  
  SMILES = dataset['smiles']
  
  data_list = []
  for smiles in SMILES:
    mol =Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    xs,z = [],[]
    for atom in mol.GetAtoms():
      x = atom_featurizer(atom)
      z.append([atom.GetAtomicNum()])
      xs.append(x)

    x = torch.tensor(xs, dtype=torch.float)
    z = torch.tensor(z)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        e = bond_featurizer(bond)
        
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]
        
    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs)
    
    y = torch.tensor(list(dataset.loc[dataset['smiles'] == smiles, label_str]))
    
    data = Data(x=x, y=y,z=z, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    data_list.append(data)
    
  return DataLoader(data_list, batch_size, shuffle=shuffle)

if __name__ == '__main__':
  dataset = pd.read_excel('../Dataset/test_data.xlsx')
  data_loader = data_process(dataset,batch_size=1,label_str='D',shuffle=True)
  for data in data_loader:
    print(data)
    break