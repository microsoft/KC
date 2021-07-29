from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def predict_single(entail_probs, text_num, hypo_type_list, thred):
    pred_label_list = []
    pid = 0
    for i in range(text_num):
        max_prob = -100.0
        max_type = None
        for hypo, cname in hypo_type_list:
            if entail_probs[pid] > max_prob:
                max_prob = entail_probs[pid]
                max_type = cname
            pid += 1
        if max_prob > thred:
            pred_label_list.append(max_type)
        else:
            pred_label_list.append('none')   
    return pred_label_list

def predict_multi(entail_probs, text_num, hypo_type_list, thred):
    pred_label_list = []
    pid = 0
    for i in range(text_num):
        pred_label_list.append([])
        for hypo, cname in hypo_type_list:
            if entail_probs[pid] > thred and (cname not in pred_label_list[-1]):
                pred_label_list[-1].append(cname)
            pid += 1
        if len(pred_label_list[-1]) == 0:
            pred_label_list[-1].append('none')
    return pred_label_list

def predict_multi_baseline(entail_probs, text_num, hypo_type_list):
    pred_label_list = []
    pid = 0
    for i in range(text_num):
        pred_label_list.append([])
        for j in range(0, len(hypo_type_list), 2):
            hypo, cname = hypo_type_list[j]
            assert "[MASK]" in hypo_type_list[j + 1][0]
            assert hypo_type_list[j + 1][1] == cname
            if entail_probs[pid] > entail_probs[pid + 1] and (cname not in pred_label_list[-1]):
                pred_label_list[-1].append(cname)
            pid += 2
        if len(pred_label_list[-1]) == 0:
            pred_label_list[-1].append('none')
    return pred_label_list    

def cal_acc(gold_label_list, pred_label_list):
    return accuracy_score(gold_label_list, pred_label_list)

#for multi-label dataset, situation
def cal_wf1(pred_label_list, gold_label_list, classes):
    text_num = len(pred_label_list)
    gold_array = np.zeros((text_num, len(classes)), dtype=int)
    pred_array = np.zeros((text_num, len(classes)), dtype=int)
    if isinstance(gold_label_list[0], list):
        for i in range(text_num):
            for cname in pred_label_list[i]:
                pred_array[i, classes.index(cname)] = 1
            for cname in gold_label_list[i]:
                gold_array[i, classes.index(cname)] = 1  
    else:
        for i in range(text_num):
            cname = pred_label_list[i]
            pred_array[i, classes.index(cname)] = 1
            cname = gold_label_list[i]
            gold_array[i, classes.index(cname)] = 1  
    wf1 = 0.0
    tot_weight = 0
    for i in range(len(classes)):
        f1 = f1_score(gold_array[:, i], pred_array[:, i], pos_label=1, average='binary')
        weight = sum(gold_array[:, i])
        wf1 += weight * f1
        tot_weight += weight
    wf1 = wf1 / tot_weight
    return wf1

