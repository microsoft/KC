# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
import numpy as np


def compute_all_scores(id_scores, ood_scores, output_dir):
    """
        It is supposed that `ood_scores` are generally *higher* than `id_scores`.
    """
    plot_dist(ood_scores, id_scores, fn=f'{output_dir}/dis_ID_neg_OOD_pos.png')

    res = {}
    
    res['AUROC'] = compute_auroc(neg_examples=id_scores, pos_examples=ood_scores, fn_curve=f'{output_dir}/AUROC_ID_neg_OOD_pos.png')
    # print(f'AUROC: {auroc_score}') # confirmed

    res['AUPR-OUT'] = compute_aupr(neg_examples=id_scores, pos_examples=ood_scores, fn_curve=f'{output_dir}/AUPR_ID_neg_OOD_pos.png')
    # print(f'AUPR-OUT: {aupr_out_score}') # confirmed

    id_scores_ = [-s for s in id_scores]
    ood_scores_ = [-s for s in ood_scores]
    res['AUPR-IN'] = compute_aupr(neg_examples=ood_scores_, pos_examples=id_scores_, fn_curve=f'{output_dir}/AUPR_ID_pos_OOD_neg.png')
    # print(f'AUPR-IN: {aupr_in_score}') # confirmed

    res['FAR95-FPR'] = compute_far(neg_examples=id_scores, pos_examples=ood_scores, target_pos_recall=0.95, definition='FPR')
    # print(f'FAR95-FPR: {far95_FPR}') # confirmed

    res['FAR95-FDR'] = compute_far(neg_examples=id_scores, pos_examples=ood_scores, target_pos_recall=0.95, definition='FDR')
    # print(f'FAR95-FDR: {far95_FDR}')

    res['DER'] = compute_der(neg_examples=id_scores, pos_examples=ood_scores, target_pos_recall=0.95)
    # print(f'DER: {fdr}')

    res['TNR'] = compute_tnr(neg_examples=id_scores, pos_examples=ood_scores, target_pos_recall=0.95)
    # print(f'TNR: {tnr}') # confirmed

    for k, v in res.items():
        print(f'{k}:\t\t{v:.4f}')
    
    return res

def compute_auroc(neg_examples, pos_examples, normalize=False, fn_curve=None):
    """
        Compute the area under the ROC curve.
        It is supposed that `pos_examples` have *higher scores* than `neg_examples`.
        ROC curve:
            x-axis: fpr - False positive rate
            y-axis: tpr - True postive rate
        Input:
            fn_curve:       str. If None, will not return the ROC curve. Else save the ROC curve to `fn_curve`.
        Return:
            AUROC score.
    """
    # plot_dist(positive=id_scores, negative=ood_scores, fn='ID_neg_OOD_pos.png')
    y = np.concatenate((np.zeros_like(neg_examples), np.ones_like(pos_examples)))
    scores = np.concatenate((neg_examples, pos_examples))
    if normalize:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    if fn_curve is not None:
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        plot_roc(fpr, tpr, roc_auc, fn=fn_curve)
        return 100 * roc_auc
    else:
        return 100 * roc_auc_score(y, scores)

def compute_aupr(neg_examples, pos_examples, normalize=False, fn_curve=None):
    """
        Compute the area under the Precision-Recall curve.
        It is supposed that `pos_examples` have *higher scores* than `neg_examples`.
        PR curve:
            x-axis: recall (TPR) - true positive rate: TP / (TP + FN)
            y-axis: precision - TP / (TP + FP)
        Input:
            fn_curve:       str. If None, will not return the ROC curve. Else save the ROC curve to `fn_curve`.
        Return:
            AUROC score.
    """
    y = np.concatenate((np.zeros_like(neg_examples), np.ones_like(pos_examples)))
    scores = np.concatenate((neg_examples, pos_examples))
    if normalize:
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    if fn_curve is not None:
        p, r, _ = precision_recall_curve(y, scores)
        aupr = auc(r, p)
        plot_pr(r, p, aupr, fn=fn_curve)
        return 100 * aupr

def compute_far(neg_examples, pos_examples, target_pos_recall=0.95, definition='FPR'):
    """
        Compute the False Alarm Rate @ `target_pos_recall` recall.
        It is supposed that `pos_examples` have *higher scores* than `neg_examples`.
        It measures the `definition` at `target_pos_recall` positive recall.
            (Types of Out-of-Distribution Texts and How to Detect Them. 2021 EMNLP.)
            (https://anssi-fr.github.io/SecuML/miscellaneous.detection_perf.html)
        Input:
            definition: choice of ['FPR', 'FDR']
                FPR (false positive rate): FP / (FP + TN)
                FDR (false discovery rate): FP / (FP + TP)
        Return:
            FAR score.
    """
    pos_examples.sort()
    threshold = pos_examples[int(len(pos_examples) * (1-target_pos_recall))]
    if isinstance(neg_examples, list):
        neg_examples = np.array(neg_examples)
    
    n_FP = sum(neg_examples > threshold)
    if definition == 'FPR':
        return 100 * n_FP / len(neg_examples)
    elif definition == 'FDR':
        return  100 * n_FP / (n_FP + len(pos_examples) * target_pos_recall)
    else:
        raise ValueError(f'Invalid `definition` input: {definition}')

def compute_der(neg_examples, pos_examples, target_pos_recall=0.95):
    """
        Compute the Detection Error Rate @ @ `target_pos_recall` recall (TPR).
        It is supposed that `pos_examples` have *higher scores* than `neg_examples`.
        DER = p_neg x FPR + p_pos X FNR
            = N_neg / (N_neg + N_pos) x FPR + N_pos / (N_neg + N_pos) x FNR.
        where
            FPR (false positive rate): FP / (FP + TN)
            FNR (false negative rate): FN / (FN + TP) = 1 - TPR = 1 - TP / (TP + FN)
    """
    pos_examples.sort()
    threshold = pos_examples[int(len(pos_examples) * (1-target_pos_recall))]
    if isinstance(neg_examples, list):
        neg_examples = np.array(neg_examples)
    
    n_neg, n_pos = len(neg_examples), len(pos_examples)
    p_neg, p_pos = n_neg / (n_neg + n_pos), n_pos / (n_neg + n_pos)

    n_FP = sum(neg_examples > threshold)

    return 100 * (p_neg * (n_FP / n_neg) + p_pos * (1 - target_pos_recall))
    # return 0.5 * (n_FP / n_neg) + 0.5 * (1 - target_pos_recall)

def compute_tnr(neg_examples, pos_examples, target_pos_recall=0.95):
    """
        Compute True Negative Rate @ @ `target_pos_recall` recall (TPR).
        It is supposed that `pos_examples` have *higher scores* than `neg_examples`.
        TNR = TN / (TN + FP)
    """
    pos_examples.sort()
    threshold = pos_examples[int(len(pos_examples) * (1-target_pos_recall))]
    if isinstance(neg_examples, list):
        neg_examples = np.array(neg_examples)

    n_TN = sum(neg_examples < threshold)
    return 100 * n_TN / len(neg_examples)


def plot_dist(positive, negative, fn):
    import matplotlib.pyplot as plt

    plt.hist(positive, 
            bins=50,
            alpha=0.5,
            label='positive')
    
    plt.hist(negative, 
            bins=50,
            alpha=0.5,
            label='negative')
    
    plt.legend(loc='upper left')
    plt.title('Overlapping')
    plt.savefig(fn)
    plt.close()

def plot_roc(fpr, tpr, roc_auc, fn):
    import matplotlib.pyplot as plt
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label="ROC curve (area = %0.8f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig(fn)
    plt.close()

def plot_pr(r, p, aupr, fn):
    import matplotlib.pyplot as plt
    plt.figure()
    lw = 2
    plt.plot(
        r,
        p,
        color="darkorange",
        lw=2,
        label="Precision-Recall curve (area = %0.8f)" % aupr,
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig(fn)
    plt.close()