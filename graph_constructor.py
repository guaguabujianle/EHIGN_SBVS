# %%
import os
import pickle
from glob import glob
import numpy as np
import argparse

import torch 
import dgl
from torch.utils.data import DataLoader

from scipy.spatial import distance_matrix
import networkx as nx
import multiprocessing
from itertools import repeat

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils import *

import warnings
warnings.filterwarnings('ignore')

# %%

def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def edge_features(mol, graph):
    geom = mol.GetConformers()[0].GetPositions()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        angles_ijk = []
        areas_ijk = []
        dists_ik = []
        for neighbor in mol.GetAtomWithIdx(j).GetNeighbors():
            k = neighbor.GetIdx() 
            if mol.GetBondBetweenAtoms(j, k) is not None and i != k:
                # geometrical features
                vector1 = geom[j] - geom[i]
                vector2 = geom[k] - geom[i]
                # angle between two vectors
                angles_ijk.append(angle(vector1, vector2))
                # area enclosed by two vectors
                areas_ijk.append(area_triangle(vector1, vector2))
                # distance between two vectors
                dists_ik.append(cal_dist(geom[i], geom[k]))

        angles_ijk = np.array(angles_ijk) if angles_ijk != [] else np.array([0.])
        areas_ijk = np.array(areas_ijk) if areas_ijk != [] else np.array([0.])
        dists_ik = np.array(dists_ik) if dists_ik != [] else np.array([0.])
        dist_ij1 = cal_dist(geom[i], geom[j], ord=1)
        dist_ij2 = cal_dist(geom[i], geom[j], ord=2)
        # length = 11
        geom_feats = [
            angles_ijk.max()*0.1,
            angles_ijk.sum()*0.01,
            angles_ijk.mean()*0.1,
            areas_ijk.max()*0.1,
            areas_ijk.sum()*0.01,
            areas_ijk.mean()*0.1,
            dists_ik.max()*0.1,
            dists_ik.sum()*0.01,
            dists_ik.mean()*0.1,
            dist_ij1*0.1,
            dist_ij2*0.1,
        ]

        bond_type = bond.GetBondType()
        basic_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]

        graph.add_edge(i, j, feats=torch.tensor(basic_feats+geom_feats).float())

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    edge_features(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v, feats in graph.edges(data=True)]).T
    edge_attr = torch.stack([feats['feats'] for u, v, feats in graph.edges(data=True)])

    return x, edge_index, edge_attr

def geom_feat(pos_i, pos_j, pos_k, angles_ijk, areas_ijk, dists_ik):
    vector1 = pos_j - pos_i
    vector2 = pos_k - pos_i
    angles_ijk.append(angle(vector1, vector2))
    areas_ijk.append(area_triangle(vector1, vector2))
    dists_ik.append(cal_dist(pos_i, pos_k))

def geom_feats(pos_i, pos_j, angles_ijk, areas_ijk, dists_ik):
    angles_ijk = np.array(angles_ijk) if angles_ijk != [] else np.array([0.])
    areas_ijk = np.array(areas_ijk) if areas_ijk != [] else np.array([0.])
    dists_ik = np.array(dists_ik) if dists_ik != [] else np.array([0.])
    dist_ij1 = cal_dist(pos_i, pos_j, ord=1)
    dist_ij2 = cal_dist(pos_i, pos_j, ord=2)
    # length = 11
    geom = [
        angles_ijk.max()*0.1,
        angles_ijk.sum()*0.01,
        angles_ijk.mean()*0.1,
        areas_ijk.max()*0.1,
        areas_ijk.sum()*0.01,
        areas_ijk.mean()*0.1,
        dists_ik.max()*0.1,
        dists_ik.sum()*0.01,
        dists_ik.mean()*0.1,
        dist_ij1*0.1,
        dist_ij2*0.1,
    ]

    return geom

def inter_graph(ligand, pocket, dis_threshold = 5.):
    graph_l2p = nx.DiGraph()
    graph_p2l = nx.DiGraph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        # ligand to pocket
        ks = node_idx[0][node_idx[1] == j]
        angles_ijk = []
        areas_ijk = []
        dists_ik = []
        for k in ks:
            if k != i:
                geom_feat(pos_l[i], pos_p[j], pos_l[k], angles_ijk, areas_ijk, dists_ik)
        geom = geom_feats(pos_l[i], pos_p[j], angles_ijk, areas_ijk, dists_ik)
        bond_feats = torch.FloatTensor(geom)
        graph_l2p.add_edge(i, j, feats=bond_feats)

        # pocket to ligand
        ks = node_idx[1][node_idx[0] == i]
        angles_ijk = []
        areas_ijk = []
        dists_ik = []
        for k in ks:
            if k != j:
                geom_feat(pos_p[j], pos_l[i], pos_p[k], angles_ijk, areas_ijk, dists_ik)     
        geom = geom_feats(pos_p[j], pos_l[i], angles_ijk, areas_ijk, dists_ik)
        bond_feats = torch.FloatTensor(geom)
        graph_p2l.add_edge(j, i, feats=bond_feats)
    
    edge_index_l2p = torch.stack([torch.LongTensor((u, v)) for u, v, feats in graph_l2p.edges(data=True)]).T
    edge_attr_l2p = torch.stack([feats['feats'] for u, v, feats in graph_l2p.edges(data=True)])

    edge_index_p2l = torch.stack([torch.LongTensor((u, v)) for u, v, feats in graph_p2l.edges(data=True)]).T
    edge_attr_p2l = torch.stack([feats['feats'] for u, v, feats in graph_p2l.edges(data=True)])

    return (edge_index_l2p, edge_attr_l2p), (edge_index_p2l, edge_attr_p2l)

# %%
def mols2graphs(complex_path, label, save_path, dis_threshold=5.0):
    try:
        with open(complex_path, 'rb') as f:
            pocket, ligand  = pickle.load(f)

        # the distance threshold to determine the interaction between ligand atoms and protein atoms
        atom_num_l = ligand.GetNumAtoms()
        atom_num_p = pocket.GetNumAtoms()

        x_l, edge_index_l, edge_attr_l = mol2graph(ligand)
        x_p, edge_index_p, edge_attr_p = mol2graph(pocket)
        (edge_index_l2p, edge_attr_l2p), (edge_index_p2l, edge_attr_p2l) = inter_graph(ligand, pocket, dis_threshold=dis_threshold)

        graph_data = {
            ('ligand', 'intra_l', 'ligand') : (edge_index_l[0], edge_index_l[1]),
            ('pocket', 'intra_p', 'pocket') : (edge_index_p[0], edge_index_p[1]),
            ('ligand', 'inter_l2p', 'pocket') : (edge_index_l2p[0], edge_index_l2p[1]),
            ('pocket', 'inter_p2l', 'ligand') : (edge_index_p2l[0], edge_index_p2l[1])
        }
        g = dgl.heterograph(graph_data, num_nodes_dict={"ligand":atom_num_l, "pocket":atom_num_p})
        g.nodes['ligand'].data['h'] = x_l
        g.nodes['pocket'].data['h'] = x_p
        g.edges['intra_l'].data['e'] = edge_attr_l
        g.edges['intra_p'].data['e'] = edge_attr_p
        g.edges['inter_l2p'].data['e'] = edge_attr_l2p
        g.edges['inter_p2l'].data['e'] = edge_attr_p2l

        if torch.any(torch.isnan(edge_attr_l)) or torch.any(torch.isnan(edge_attr_p)):
            status = False
        else:
            status = True
    except:
        g = None
        status = False

    if status:
        torch.save((g, torch.FloatTensor([label])), save_path)
        return save_path    
    else:
        print(f"Failed to process {complex_path}")
        return None


# %%
def collate_fn(data_batch):
    g, label = map(list, zip(*data_batch))
    bg = dgl.batch(g)
    y = torch.cat(label, dim=0)

    return bg, y

class GraphDataset(object):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, mol_paths, labels, graph_paths=None, dis_threshold=5.0, num_process=48):
        self.mol_paths = mol_paths
        self.dis_threshold = dis_threshold
        self.labels = labels
        self.num_process = num_process
        self.graph_paths = graph_paths
        self._pre_process()

    def _pre_process(self):
        mol_paths = self.mol_paths
        labels = self.labels

        if self.graph_paths == None:
            dis_thresholds = repeat(self.dis_threshold, len(mol_paths))
            graph_path_list = [mol_path.replace(".dat", "-EHIGN.dgl") for mol_path in mol_paths]

            print('Generate complex graph...')
            pool = multiprocessing.Pool(self.num_process)
            ret = pool.starmap(mols2graphs,
                            zip(mol_paths, labels, graph_path_list, dis_thresholds))
            pool.close()
            pool.join()


    def __getitem__(self, idx):
        return torch.load(self.graph_paths[idx])

    def __len__(self):
        return len(self.graph_paths)

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='data directory')
    args = parser.parse_args()
    data_root = args.data_root

    # use FEN1_5fv7 as an example
    for complex_ids in ['FEN1_5fv7']:
    # for complex_ids in ['FEN1_5fv7', 'ALDH1_4x4l', 'GBA_2v3e', 'KAT2A_5h84', 'MAPK1_2ojg', 'PKM2_3gr4', 'VDR_3a2j']:
        complex_dir = os.path.join(data_root, complex_ids)

        train_dir = os.path.join(complex_dir, 'train')
        val_dir = os.path.join(complex_dir, 'val')

        train_mol_paths = glob(os.path.join(train_dir , "*.dat"))
        train_labels = [1 if 'actives' in tp else 0 for tp in train_mol_paths]
        print('train positive: ', np.sum(train_labels))

        val_mol_paths = glob(os.path.join(val_dir , "*.dat"))
        val_labels = [1 if 'actives' in tp else 0 for tp in val_mol_paths]
        print('val positive: ', np.sum(val_labels))

        # generate graphs, using 48 threads by default.
        train_set = GraphDataset(train_mol_paths, train_labels)
        val_set = GraphDataset(val_mol_paths, val_labels)

        # save graph paths to pickle file
        train_graph_paths = glob(os.path.join(train_dir , "*.dgl"))
        train_labels = [1 if 'actives' in tp else 0 for tp in train_graph_paths]
        val_graph_paths = glob(os.path.join(val_dir , "*.dgl"))
        val_labels = [1 if 'actives' in tp else 0 for tp in val_graph_paths]

        write_pickle(os.path.join(data_root, f"train_graph_paths_{complex_ids}.pkl"), (train_graph_paths, train_labels))
        write_pickle(os.path.join(data_root, f"val_graph_paths_{complex_ids}.pkl"), (val_graph_paths, val_labels))


# %%
