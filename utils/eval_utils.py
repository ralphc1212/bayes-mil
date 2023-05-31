import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
from models.model_bmil import bMIL_model_dict
from models.model_bmil import get_ard_reg_vdo
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    

    if args.model_type.startswith('bmil'):
        model = bMIL_model_dict[args.model_type.split('-')[1]](**model_dict)
        bayes_args = [get_ard_reg_vdo, 1e-5]
        if 'spvis' in args.model_type.split('-'):
            bayes_args.append('spvis')
        elif 'vis' in args.model_type.split('-'):
            bayes_args.append('vis')
        elif 'enc' in args.model_type.split('-'):
            bayes_args.append('enc')

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)
    model.relocate()
    model.eval()
    if args.model_type.startswith('bmil'):
        return model, bayes_args
    else:
        return model

def eval(dataset, args, ckpt_path):
    model, bayes_args = initiate_model(args, ckpt_path)

    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args, bayes_args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args, bayes_args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (slide_id, data, label, coords, width, height) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        coords = coords.cpu().numpy()
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            if 'vis' in bayes_args :
                logits, Y_prob, Y_hat, _, A = model(data, validation=True)
            elif 'enc' in bayes_args :
                logits, Y_prob, Y_hat, _, A = model(data, validation=True)
            elif 'spvis' in bayes_args:
                logits, Y_prob, Y_hat, _, A = model(data, coords, height[0], width[0], validation=True)

        acc_logger.log(Y_hat, label)

        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})

        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else:
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger
