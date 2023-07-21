# %%
import torch.nn as nn
from CIGConv import CIGConv
from NIGConv import NIGConv
from HGC import HeteroGraphConv
import dgl

class DTIPredictor(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size, layer_num=3):
        super(DTIPredictor, self).__init__()

        self.convs = nn.ModuleList()

        for _ in range(layer_num):
            convl = CIGConv(hidden_feat_size, hidden_feat_size)
            convp = CIGConv(hidden_feat_size, hidden_feat_size)
            convlp = NIGConv(hidden_feat_size, hidden_feat_size, feat_drop=0.1)
            convpl = NIGConv(hidden_feat_size, hidden_feat_size, feat_drop=0.1)
            conv = HeteroGraphConv(
                    {
                        'intra_l' : convl,
                        'intra_p' : convp,
                        'inter_l2p' : convlp,
                        'inter_p2l': convpl
                    }
                )
            self.convs.append(conv)

        self.lin_node_l = nn.Linear(node_feat_size, hidden_feat_size)
        self.lin_node_p = nn.Linear(node_feat_size, hidden_feat_size)
        self.lin_edge_ll = nn.Linear(edge_feat_size, hidden_feat_size)
        self.lin_edge_pp = nn.Linear(edge_feat_size, hidden_feat_size)

        self.lin_edge_lp = nn.Linear(11, hidden_feat_size)
        self.lin_edge_pl = nn.Linear(11, hidden_feat_size)

        self.fc = FC(hidden_feat_size, hidden_feat_size, 3, 0.1, 1)

    def forward(self, bg):
        atom_feats = bg.ndata['h']
        bond_feats = bg.edata['e']

        atom_feats = {
            'ligand':self.lin_node_l(atom_feats['ligand']),
            'pocket':self.lin_node_p(atom_feats['pocket'])
        }
        bond_feats = {
            ('ligand', 'intra_l', 'ligand'):self.lin_edge_ll(bond_feats[('ligand', 'intra_l', 'ligand')]),
            ('pocket', 'intra_p', 'pocket'):self.lin_edge_pp(bond_feats[('pocket', 'intra_p', 'pocket')]),       
            ('ligand', 'inter_l2p', 'pocket'):self.lin_edge_lp(bond_feats[('ligand', 'inter_l2p', 'pocket')]),    
            ('pocket', 'inter_p2l', 'ligand'):self.lin_edge_pl(bond_feats[('pocket', 'inter_p2l', 'ligand')]),        
        }

        bg.edata['e'] = bond_feats

        rsts = atom_feats
        for conv in self.convs:
            rsts = conv(bg, rsts)

        bg.nodes['ligand'].data['h'] = rsts['ligand']
        bg.nodes['pocket'].data['h'] = rsts['pocket']

        ligand_pooled = dgl.readout_nodes(bg, 'h', ntype='ligand')
        pocket_pooled = dgl.readout_nodes(bg, 'h', ntype='pocket')

        logits = self.fc(ligand_pooled + pocket_pooled)

        return logits

class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h

            

# %%
