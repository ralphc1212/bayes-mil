import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits

from models.model_bmil import bMIL_model_dict
from models.model_bmil import  get_ard_reg_vdo

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

from torch.optim.lr_scheduler import ReduceLROnPlateau

N_SAMPLES = 16

import torch.nn as nn
class ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

eceloss = ECELoss()


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type == 'clam' and args.subtyping:
        model_dict.update({'subtyping': True})
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    bayes_args = None


    if args.model_type.startswith('bmil'):
        model = bMIL_model_dict[args.model_type.split('-')[1]](**model_dict)
        bayes_args = [get_ard_reg_vdo, 1e-8, 1e-6]
        if 'vis' in args.model_type.split('-'):
            bayes_args.append('vis')
        elif 'spvis' in args.model_type.split('-'):
            bayes_args.append('spvis')
        elif 'enc' in args.model_type.split('-'):
            bayes_args.append('enc')
    else:
        raise NotImplementedError

    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None

    print('Done!')

    # stochastic = (bayes_reg != None)

    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, bayes_args)
        stop, val_loss = validate(cur, epoch, model, val_loader, args.n_classes,
            early_stopping, writer, loss_fn, args.results_dir, bayes_args)
        scheduler.step(val_loss)

        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, val_ece_loss, _ = summary(model, val_loader, args.n_classes, bayes_args=['spvis'])
    print('Val error: {:.4f}, ROC AUC: {:.4f}, ece loss : {:.4f}'.format(val_error, val_auc, val_ece_loss))

    results_dict, test_error, test_auc, test_ece_loss, acc_logger = summary(model, test_loader, args.n_classes, bayes_args=['spvis'])
    print('Test error: {:.4f}, ROC AUC: {:.4f}, ece loss : {:.4f}'.format(test_error, test_auc, test_ece_loss))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/val_ece_loss', val_ece_loss, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.add_scalar('final/test_ece_loss', test_ece_loss, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error, test_ece_loss, val_ece_loss


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, bayes_args=None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    ece_losses = []
    train_error = 0.

    print('\n')
    for batch_idx, (slide_id, data, label, coords, width, height) in enumerate(loader):
        data, label = data.to(device), label.to(device)

        if 'enc' in bayes_args:
            logits, Y_prob, Y_hat, kl_div, _, _ = model(data, slide_label=label)
        elif 'spvis' in bayes_args:
            coords = coords.cpu().numpy()
            logits, Y_prob, Y_hat, kl_div, _, _ = model(data, coords, height[0], width[0],  slide_label=label)
        elif 'vis' in bayes_args:
            logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        ece_losses.append(eceloss(logits, label).cpu().item())

        
        if 'vis' in bayes_args:
            loss += bayes_args[1] * bayes_args[0](model)

        elif 'enc' in bayes_args :
            kl_model = bayes_args[0](model)
            kl_data = kl_div[0]
            loss += bayes_args[1] * kl_model + bayes_args[2] * kl_data
        elif 'spvis' in bayes_args :
            kl_model = bayes_args[0](model)
            kl_div = kl_div.reshape(-1)
            kl_data = kl_div[0]
            loss += bayes_args[1] * kl_model + bayes_args[2] * kl_data
            
        loss_value = loss.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
            # print(model.state_dict()['attn_thres_r'])

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    ece_loss = np.mean(ece_losses)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, ece_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, ece_loss,
                                                                                        train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/ece_loss', ece_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def validate(cur, epoch, model, loader, n_classes, early_stopping = None,
             writer = None, loss_fn = None, results_dir=None,  bayes_args=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bayes_args and ('vis' in bayes_args or 'spvis' in bayes_args or 'enc' in bayes_args):
        model.train()
    else:
        model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    ece_losses = []
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    slide_model_uncertainty = []
    slide_data_uncertainty = []

    attention_model_uncertainty = []
    attention_data_uncertainty = []

    with torch.no_grad():
        for batch_idx, (slide_id, data, label, coords, width, height) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)
            coords = coords.cpu().numpy()

            if bayes_args and ('vis' in bayes_args or 'spvis' in bayes_args or 'enc' in bayes_args):
                out_prob = 0
                out_atten = 0
                out_logits = 0
                # EXTRACT DATA UNCERTAINTY: vis_data = 0 

                Y_hats = []
                ens_prob = []
                ens_atten = []
                for i in range(N_SAMPLES):
                    if 'vis' in bayes_args or 'enc' in bayes_args:
                        logits, Y_prob, Y_hat, _, A = model(data, validation=True)
                    elif 'spvis' in bayes_args:
                        logits, Y_prob, Y_hat, _, A = model(data, coords, height[0], width[0], validation=True)
                    out_prob += Y_prob
                    out_atten += A.detach()
                    out_logits += logits

                    Y_hats.append(Y_hat)
                    ens_prob.append(torch.sum(- Y_prob * torch.log(Y_prob)).item())
                    if 'vis' in bayes_args or 'spvis' in bayes_args or 'enc' in bayes_args:
                        A = A.t()
                        A = torch.cat([A, 1 - A], dim = 1)
                        ens_atten.append((- A * torch.log(A)).sum(dim = 1).mean().item())
                        # EXTRACT DATA UNCERTAINTY: store the vector vis_data += (- A * torch.log(A)).sum(dim = 1)
                    else:
                        ens_atten.append(torch.sum(- A * torch.log(A)).item())

                out_prob /= N_SAMPLES
                out_atten /= N_SAMPLES
                out_logits /= N_SAMPLES
                # EXTRACT DATA UNCERTAINTY: vis_data /= N_SAMPLES
                # vis_data size: [number of patches, 1]

                out_ens_prob = torch.sum(- out_prob * torch.log(out_prob)).item()
                out_atten = out_atten.t()
                out_atten = torch.cat([out_atten, 1 - out_atten], dim = 1)
                out_ens_atten = (- out_atten * torch.log(out_atten)).sum(dim = 1).mean().item()
                # EXTRACT TOTAL UNCERTAINTY: vis_total = (- out_atten * torch.log(out_atten)).sum(dim = 1)
                # vis_total size: [number of patches, 1]

                # EXTRACT TOTAL UNCERTAINTY: vis_total - vis_data

                ens_prob = np.mean(ens_prob)
                ens_atten = np.mean(ens_atten)
                Y_hat = torch.mode(torch.cat(Y_hats, dim=1))[0]

                slide_model_uncertainty.append(out_ens_prob - ens_prob)
                slide_data_uncertainty.append(ens_prob)

                attention_model_uncertainty.append(out_ens_atten - ens_atten)
                attention_data_uncertainty.append(ens_atten)

            else:
                logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)
            ece_losses.append(eceloss(logits, label).cpu().item())

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    ece_loss = np.mean(ece_losses)
    val_loss /= len(loader)
    if bayes_args:
        slide_model_uncertainty = np.mean(slide_model_uncertainty)
        slide_data_uncertainty = np.mean(slide_data_uncertainty)
        attention_model_uncertainty = np.mean(attention_model_uncertainty)
        attention_data_uncertainty = np.mean(attention_data_uncertainty)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/ece_loss', ece_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, ece_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, ece_loss, val_error, auc))
    print('\nVal Set, slide_model_unc: {:.4f}, attn_model_unc: {:.4f}, slide_data_unc: {:.4f}, attn_data_unc: {:.4f}'
        .format(slide_model_uncertainty, attention_model_uncertainty, slide_data_uncertainty, attention_data_uncertainty))

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True, val_loss

    return False, val_loss


def summary(model, loader, n_classes, bayes_args=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    ece_losses = []
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

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
        ece_losses.append(eceloss(logits, label).cpu().item())
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)
    ece_loss = np.mean(ece_losses)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, ece_loss, acc_logger

