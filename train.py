
# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import time
import math
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from graph_constructor import GraphDataset, collate_fn
from utils import *
from EHIGN import DTIPredictor
from metrics import *
from config.config_dict import *
from log.train_logger import *

import warnings
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

# %%
def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()
    
    pred_list = []
    pred_cls_list = []
    label_list = []
    for data in dataloader:
        bg, label = data
        bg, label = bg.to(device), label.to(device)

        with torch.no_grad():
            pred = model(bg).view(-1)
            loss = criterion(pred, label)

            pred_prob = torch.sigmoid(pred)
            pred_list.append(pred_prob.view(-1).detach().cpu().numpy())
            pred_cls = pred_prob > 0.5
            pred_cls_list.append(pred_cls.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred = np.concatenate(pred_list, axis=0)
    pred_cls = np.concatenate(pred_cls_list, axis=0)
    label = np.concatenate(label_list, axis=0).astype("int")

    acc = accuracy(label, pred_cls)
    ap = average_precision_score(label, pred)
    auc = roc_auc_score(label, pred)
    sen = sensitive(label, pred_cls)
    spec = specificity(label, pred_cls) 

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    bedroc = bedroc_score(label, pred, alpha=80.5)

    model.train()

    return epoch_loss, acc, sen, spec, ap, bedroc, auc

def train():
    # use FEN1_5fv7 as an example
    for target_name in ['FEN1_5fv7']:
    # for target_name in ['FEN1_5fv7', 'ALDH1_4x4l', 'GBA_2v3e', 'KAT2A_5h84', 'MAPK1_2ojg', 'PKM2_3gr4', 'VDR_3a2j']:
        cfg = 'TrainConfig'
        config = Config(cfg)

        args = config.get_config()
        args['target'] = target_name
        target = args.get("target")
        args['mark'] = target

        save_model = args.get("save_model")
        batch_size = args.get("batch_size")
        data_root = args.get('data_root')
        epochs = args.get('epochs')
        steps_per_epoch = args.get('steps_per_epoch')
        early_stop_epoch = args.get("early_stop_epoch")
        create = args.get("create")
        logger = TrainLogger(args, cfg, create)

        train_paths, train_labels = read_pickle(os.path.join(data_root, f"train_graph_paths_{target}.pkl"))
        val_paths, val_labels = read_pickle(os.path.join(data_root, f"val_graph_paths_{target}.pkl"))
        weight = np.sum(np.array(train_labels) == 0) / np.sum(np.array(train_labels) == 1) * 0.3
        sample_weight = [weight if label == 1 else 1 for label in train_labels]
        sampler = WeightedRandomSampler(sample_weight, len(train_labels))

        train_set = GraphDataset(mol_paths=None, labels=None, graph_paths=train_paths)
        train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=8, sampler=sampler)

        val_set = GraphDataset(mol_paths=None, labels=None, graph_paths=val_paths)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)

        logger.info(f"train data: {len(train_set)}")
        logger.info(f"val data: {len(val_set)}")

        device = torch.device('cuda:0')
        model =  DTIPredictor(node_feat_size=35, edge_feat_size=17, hidden_feat_size=256)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=1e-6)
        criterion = nn.BCEWithLogitsLoss()

        running_loss = AverageMeter()
        running_best_bedroc = BestMeter("max")

        num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
        global_step = 0
        global_epoch = 0
        break_flag = False

        model.train()
        for i in range(num_iter):
            if break_flag:
                break

            for data in train_loader:

                global_step += 1

                bg, label = data
                bg, label = bg.to(device), label.to(device)

                pred = model(bg).view(-1)
                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), label.size(0)) 

                if global_step % steps_per_epoch == 0:
                    global_epoch += 1

                    epoch_loss = running_loss.get_average()
                    running_loss.reset()

                    val_loss, val_acc, val_sen, val_spec, val_ap, val_bedroc, val_auc = val(model, criterion, val_loader, device)
            
                    msg = "epoch-%d, train_loss-%.4f, val_loss-%.4f, val_acc-%.4f, val_sen-%.4f, val_spec-%.4f, val_ap-%.4f, val_bedroc-%.4f, val_auc-%.4f" \
                        % (global_epoch, epoch_loss, val_loss, val_acc, val_sen, val_spec, val_ap, val_bedroc, val_auc)
                    if create:
                        logger.info(msg)
                    else:
                        print(msg)

                    if save_model:
                        msg = "epoch-%d, train_loss-%.4f, val_loss-%.4f, val_acc-%.4f, val_sen-%.4f, val_spec-%.4f, val_ap-%.4f, val_bedroc-%.4f, val_auc-%.4f" % (global_epoch, epoch_loss, val_loss, val_acc, val_sen, val_spec, val_ap, val_bedroc, val_auc)
                        save_model_dict(model, logger.get_model_dir(), msg)

                    if val_bedroc > running_best_bedroc.get_best():
                        running_best_bedroc.update(val_bedroc)
                        if save_model:
                            msg = "best_model"
                            save_model_dict(model, logger.get_model_dir(), msg)
                    else:
                        count = running_best_bedroc.counter()
                        if count > early_stop_epoch:
                            running_best_bedroc = running_best_bedroc.get_best()
                            msg = "best_bedroc: %.4f" % running_best_bedroc
                            logger.info(f"early stop in epoch {global_epoch}")
                            logger.info(msg)
                            break_flag = True
                            break

            time.sleep(1)

if __name__ == "__main__":
    train()
# %%
