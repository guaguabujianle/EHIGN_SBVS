
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

import argparse
import pandas as pd
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from graph_constructor import GraphDataset, collate_fn
from EHIGN import DTIPredictor

from metrics import *
from utils import *

import warnings
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

# %%
def val(model, criterion, dataloader, device):
    model.eval()
    pred_dict = defaultdict(list)
    performance_dict = defaultdict(list)
    for data in dataloader:
        bg, label = data
        bg, label = bg.to(device), label.to(device)

        with torch.no_grad():
            pred = model(bg).view(-1)
            loss = criterion(pred, label)

            pred_prob = torch.sigmoid(pred)
            pred_cls = pred_prob > 0.5
            pred_prob = pred_prob.view(-1).detach().cpu().numpy().tolist()
            pred_cls = pred_cls.view(-1).detach().cpu().numpy().tolist()
            label = label.detach().cpu().numpy().tolist()

            pred_dict['pred_prob'].extend(pred_prob)
            pred_dict['pred_cls'].extend(pred_cls)
            pred_dict['label'].extend(label)

    pred_df = pd.DataFrame(pred_dict)
    pred = pred_df['pred_prob']
    pred_cls = pred_df['pred_cls']
    label = pred_df['label']
    ap = average_precision_score(label, pred)
    auc = roc_auc_score(label, pred)
    bedroc = bedroc_score(label, pred, alpha=80.5)
    enrich = enrichment_score(label, pred)

    performance_dict['auc'].append(auc)
    performance_dict['bedroc'].append(bedroc)
    performance_dict['ap'].append(ap)
    performance_dict['enrich0.001'].append(enrich[0])
    performance_dict['enrich0.005'].append(enrich[1])
    performance_dict['enrich0.01'].append(enrich[2])
    performance_dict['enrich0.05'].append(enrich[3])

    performance_df = pd.DataFrame(performance_dict)

    return performance_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='data directory')
    args = parser.parse_args()
    data_root = args.data_root

    device = torch.device('cuda:0')
    model =  DTIPredictor(node_feat_size=35, edge_feat_size=17, hidden_feat_size=256)
    criterion = nn.BCEWithLogitsLoss()

    model_dir = './model'
    # use FEN1_5fv7 as an example
    for complex_ids in ['FEN1_5fv7']:
    # for complex_ids in ['ALDH1_4x4l', 'FEN1_5fv7', 'GBA_2v3e', 'KAT2A_5h84', 'MAPK1_2ojg', 'PKM2_3gr4', 'VDR_3a2j']:
        valid_paths, _ = read_pickle(os.path.join(data_root, f"val_graph_paths_{complex_ids}.pkl"))
        val_set = GraphDataset(mol_paths=None, labels=None, graph_paths=valid_paths)
        val_loader = DataLoader(val_set, batch_size=2048, shuffle=False, collate_fn=collate_fn, num_workers=8)

        if complex_ids == 'ALDH1_4x4l':
            model_path = os.path.join(model_dir, '20230701_074948_HIGN_LIT-PCBA_fold0_ALDH1_4x4l', 'model', 'best_model.pt')
        elif complex_ids == 'FEN1_5fv7':
            model_path = os.path.join(model_dir, '20230701_040331_HIGN_LIT-PCBA_fold0_FEN1_5fv7', 'model', 'best_model.pt')
        elif complex_ids == 'GBA_2v3e':
            model_path = os.path.join(model_dir, '20230701_103906_HIGN_LIT-PCBA_fold0_GBA_2v3e', 'model', 'best_model.pt')
        elif complex_ids == 'KAT2A_5h84':
            model_path = os.path.join(model_dir, '20230701_123952_HIGN_LIT-PCBA_fold0_KAT2A_5h84', 'model', 'best_model.pt')
        elif complex_ids == 'MAPK1_2ojg':
            model_path = os.path.join(model_dir, '20230701_143527_HIGN_LIT-PCBA_fold0_MAPK1_2ojg', 'model', 'best_model.pt')
        elif complex_ids == 'PKM2_3gr4':
            model_path = os.path.join(model_dir, '20230701_151544_HIGN_LIT-PCBA_fold0_PKM2_3gr4', 'model', 'best_model.pt')
        elif complex_ids == "VDR_3a2j":
            model_path = os.path.join(model_dir, '20230701_170007_HIGN_LIT-PCBA_fold0_VDR_3a2j', 'model', 'best_model.pt')
        
        load_model_dict(model, model_path)
        model = model.to(device)

        df = val(model, criterion, val_loader, device)
        df.to_csv(f"./paper/EHIGN_LIT-PCBA_{complex_ids}_target_specific.csv", index=False)

# %%

