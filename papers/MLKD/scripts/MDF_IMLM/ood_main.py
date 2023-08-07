#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

import os 
import pdb 
import json
import math
import torch 
import random
import logging 
import argparse
import pickle
import numpy as np 
import pandas as pd 
import calculate_log as callog

from sklearn import svm 
import sklearn
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm 
from simpletransformers.classification import ClassificationModel 
from our_model import our_model
from ood_metrics import compute_all_scores

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

seed = 171

def detect(all_test_deviations,all_ood_deviations, verbose=True, normalize=True):
    average_results = {}
    for i in range(1,11):
        random.seed(i)

        validation_indices = random.sample(range(len(all_test_deviations)),int(0.1*len(all_test_deviations)))
        test_indices = sorted(list(set(range(len(all_test_deviations)))-set(validation_indices)))

        validation = all_test_deviations[validation_indices]
        test_deviations = all_test_deviations[test_indices]

        t95 = validation.mean(axis=0)+10**-7
        if not normalize:
            t95 = np.ones_like(t95)
        test_deviations = (test_deviations/t95[np.newaxis,:]).sum(axis=1)
        ood_deviations = (all_ood_deviations/t95[np.newaxis,:]).sum(axis=1)
              
        results = callog.compute_metric(-test_deviations,-ood_deviations)
        for m in results:
            average_results[m] = average_results.get(m,0)+results[m]
    
    for m in average_results:
        average_results[m] /= i
    if verbose:
        callog.print_results(average_results)
    return average_results


def detection_performance(scores, Y, outf, tag='TMP'):
    """
    Measure the detection performance
    return: detection metrics
    """
    os.makedirs(outf, exist_ok=True)
    num_samples = scores.shape[0]
    l1 = open('%s/confidence_%s_In.txt'%(outf, tag), 'w')
    l2 = open('%s/confidence_%s_Out.txt'%(outf, tag), 'w')
    y_pred = scores # regressor.predict_proba(X)[:, 1]

    for i in range(num_samples):
        if Y[i] == 0:
            l1.write("{}\n".format(-y_pred[i]))
        else:
            l2.write("{}\n".format(-y_pred[i]))
    l1.close()
    l2.close()
    results = callog.metric(outf, [tag])
    return results
    
def load_ROSTD_dataset():
    train_file = f"./HC3/hc3_train.txt"
    test_file = f"./HC3/hc3_test.txt"
    outlier_file = f"./HC3/hc3_ood.txt"
    train_df = load_ROSTD_extra_dataset(train_file, label=1)
    test_df = load_ROSTD_extra_dataset(test_file, label=1)
    ood_df = load_ROSTD_extra_dataset(outlier_file, label=0)
    return train_df, test_df, ood_df


def load_sst_dataset():
    train_df = load_extra_dataset("./dataset/sst/sst-train.txt", label=1)
    test_df = load_extra_dataset("./dataset/sst/sst-test.txt", label=1)
    ood_snli_df = load_extra_dataset("./dataset/sst/snli-dev.txt", drop_index=True, label=0)
    ood_rte_df = load_extra_dataset("./dataset/sst/rte-dev.txt", drop_index=True, label=0)
    ood_20ng_df = load_extra_dataset("./dataset/sst/20ng-test.txt", drop_index=True, label=0)
    ood_multi30k_df = load_extra_dataset("./dataset/sst/multi30k-val.txt", drop_index=True, label=0)
    ood_snli_df = ood_snli_df.sample(n=500, random_state=seed)
    ood_rte_df = ood_rte_df.sample(n=500, random_state=seed)
    ood_20ng_df = ood_20ng_df.sample(n=500, random_state=seed)
    ood_multi30k_df = ood_multi30k_df.sample(n=500, random_state=seed)
    ood_df = pd.concat([ood_snli_df, ood_rte_df, ood_20ng_df, ood_multi30k_df])
    # ood_df = ood_df.sample(n=len(test_df), random_state=seed)
    # pdb.set_trace()
    return train_df, test_df, ood_df

def load_dataset(data_name, data_type='full'):
    with open('./dataset/CLINIC150/data_full.json', 'r') as f:
        data = json.load(f)
    field = "_".join(data_name.split("_")[1:])
    dataset = data[field]
    data_df = pd.DataFrame(dataset, columns=['text', 'labels'])  # labels are not used for training
    return data_df
    

def load_ROSTD_extra_dataset(file_path="./dataset/SSTSentences.txt", drop_index=False, label=0):
    with open(file_path, 'r') as f:
        data = [ii.strip() for ii in f.readlines()]
    df = pd.DataFrame(data, columns=['text'])
    df["index"] = df.index
    df['labels'] = label
    df.rename(columns = {'sentence': 'text'}, inplace=True)
    if drop_index:
        df.drop(columns='index', inplace=True)
    df.dropna(inplace=True)
    return df

def load_extra_dataset(file_path="./dataset/SSTSentences.txt", drop_index=False, label=0):
    df = pd.read_csv(file_path, sep='\t', header=0)
    df['labels'] = label
    df.rename(columns = {'sentence': 'text'}, inplace=True)
    if drop_index:
        df.drop(columns='index', inplace=True)
    df.dropna(inplace=True)
    return df

def frequency_OOD_detect(args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    data_type = args.data_type.strip() # args.data_type.strip()
    if data_type == "sst":
        print ("Work on different datasets")
        train_df, test_df, ood_df = load_sst_dataset()
        # pdb.set_trace()
    else:
        print ("Work on different intents")
        ood_df  = load_dataset('clinc150_oos_test', data_type=data_type) 
        train_df = load_dataset('clinc150_train', data_type=data_type)
        test_df  = load_dataset('clinc150_test', data_type=data_type)

    v = TfidfVectorizer()
    train_feats = v.fit_transform(train_df['text'])
    test_feats = v.transform(test_df['text'])
    ood_feats = v.transform(ood_df['text'])

    svd = TruncatedSVD(100)
    train_feats = svd.fit_transform(train_feats)
    print ('train_feats', train_feats.shape)
    test_feats = svd.transform(test_feats)
    print ("test feats", test_feats.shape)
    ood_feats = svd.transform(ood_feats)
    print ('ood feats', ood_feats.shape)

    candidate_list = [1e-9, 1e-7, 1e-5, 1e-3, 0.1, 0.5, 0.7, 0.9]           
    # candidate_list = [1e-9, 0.001, 0.1, 0.5, 0.9]
    best_ours_AUC = 0.0 
    for k in ['linear']:
        for nuu in tqdm(candidate_list):
            c_lr = svm.OneClassSVM(nu=nuu, kernel=k, degree=2)
            c_lr.fit(train_feats)

            test_scores =c_lr.score_samples(test_feats)
            ood_scores = c_lr.score_samples(ood_feats)
            
            X_scores = np.concatenate((ood_scores, test_scores))
            ood_labels = np.ones_like(ood_scores) 
            test_labels = np.zeros_like(test_scores)
            Y_test = np.concatenate((ood_labels, test_labels))
            
            raw_results = detection_performance(X_scores, Y_test, 'feats_logs', tag='XXX')
            neg_resuls = detection_performance(-X_scores, Y_test, 'feats_logs', tag='XXX')
            if sum(raw_results["XXX"].values()) < sum(neg_resuls["XXX"].values()):
                raw_results = neg_resuls
            results = raw_results['XXX']
            # print ("results", results)
            if results['AUROC'] > best_ours_AUC:
                best_ours_AUC = results['AUROC']
                best_ours_results = results
                best_hypers = "{}-{}".format(k, nuu)
                d = {"X_scores": X_scores, "Y_test": Y_test}
        
    mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100.*best_ours_results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*best_ours_results['DTACC']), end='')
    print(' {val:6.2f}'.format(val=100.*best_ours_results['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*best_ours_results['AUOUT']), end='')
    print("best hyper %s"%(best_hypers)) 
    print("saving best model results")
    with open("./outputs/{}-{}.pkl".format(data_type, args.type), "wb") as f:
        pickle.dump(d, f)
    print('-------------------------------')


def single_layer_OOD_detect(args):
    data_type = args.data_type.strip() # args.data_type.strip()
    print("data type %s"%( data_type))
    if args.load_path:
        load_path = args.load_path
    else:
        load_path = 'bert-base-uncased' if args.model_class == 'bert' else 'roberta-base'
    if args.model_class == 'bert' or args.model_class == 'roberta':
        assert load_path is not None
        model = our_model(
        args.model_class, 
        load_path, 
        num_labels=2,
        use_cuda=True,
        cuda_device=int(args.gpu_id),
        args={'num_train_epochs': 0,
              'fp16':False,
              'n_gpu':int(args.n_gpu),
              'learning_rate': 4e-5,
              'warmup_ratio': 0.10,
              'train_batch_size': 32,
              'eval_batch_size': 32, 
              #'evaluate_during_training': False,
              #'evaluate_during_training_steps': 2000,    
              'do_lower_case': True,
              'silent':True,
              'reprocess_input_data': True, 
              'overwrite_output_dir': False,
              'output_dir': '%s_outputs/'%(data_type),
              'best_model_dir': "%s_outputs/best_model"%(data_type),
              'cache_dir': "%s_cache_dir/"%(data_type)})
    else:
        raise NotImplementedError

    if data_type == "sst":
        print ("Work on different datasets")
        train_df, test_df, ood_df = load_sst_dataset()
    else:
        print ("Work on different intents")
        ood_df = load_dataset('clinc150_oos_test', data_type=data_type) 
        train_df  = load_dataset('clinc150_train', data_type=data_type)
        test_df = load_dataset('clinc150_test', data_type=data_type)

    for layer in [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12]:
    # for layer in ['all-max', 'all-mean']:
        for use_cls in [False, True]:
            print ("--------- we are using {} layer and {} to represent sequence --------".format(layer, "[CLS]" if use_cls else "AVG"))
            ood_feats = model.get_one_layer_feature(ood_df['text'].values.tolist(), use_layer=layer, use_cls=use_cls)  # n_sample x 768 
            train_feats = model.get_one_layer_feature(train_df['text'].values.tolist(), use_layer=layer, use_cls=use_cls) 
            test_feats = model.get_one_layer_feature(test_df['text'].values.tolist(), use_layer=layer, use_cls=use_cls)  

            candidate_list = [1e-9, 1e-7, 1e-3, 0.01, 0.1]  # add 0.5 and 0.7 for slower training
            best_ours_AUC = 0.0 
            for k in ['linear']:
                for nuu in tqdm(candidate_list):
                    c_lr = svm.OneClassSVM(nu=nuu, kernel=k, degree=2)
                    c_lr.fit(train_feats)

                    test_scores = c_lr.score_samples(test_feats)
                    ood_scores = c_lr.score_samples(ood_feats)
                    
                    X_scores = np.concatenate((ood_scores, test_scores))
                    ood_labels = np.ones_like(ood_scores) 
                    test_labels = np.zeros_like(test_scores)
                    Y_test = np.concatenate((ood_labels, test_labels))

                    raw_results = detection_performance(X_scores, Y_test, 'feats_logs', tag='XXX')
                    neg_resuls = detection_performance(-X_scores, Y_test, 'feats_logs', tag='XXX')
                    if sum(raw_results["XXX"].values()) < sum(neg_resuls["XXX"].values()):
                        raw_results = neg_resuls
                    results = raw_results['XXX']
                    if results['AUROC'] > best_ours_AUC:
                        best_ours_AUC = results['AUROC']
                        best_ours_results = results
                        best_hypers = "{}-{}".format(k, nuu)
                        d = {"X_scores": X_scores, "Y_test": Y_test, "Features": np.concatenate((test_scores, ood_scores))}
                
            mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*best_ours_results['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*best_ours_results['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*best_ours_results['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*best_ours_results['AUOUT']), end='')
            print("best hyper %s"%(best_hypers)) 
            print('-------------------------------')

    # print("saving best model results")
    # with open("./outputs/{}-{}-{}-{}.pkl".format(data_type, args.method, args.model_class, layer), "wb") as f:
        # pickle.dump(d, f)
    

def MDF_OOD_detect(args, distance="mahalanobis"):
    data_type = args.data_type.strip()
    print("data type %s"%( data_type))
    if args.load_path:
        load_path = args.load_path
    else:
        load_path = 'bert-base-uncased' if args.model_class == 'bert' else 'roberta-base'
    if args.model_class == 'bert' or args.model_class == 'roberta':
        assert load_path is not None
        model = our_model(
        args.model_class, 
        load_path, 
        num_labels=2,
        use_cuda=True,
        cuda_device=int(args.gpu_id),
        args={'num_train_epochs': 0,
              'fp16':False,
              'n_gpu':int(args.n_gpu),
              'learning_rate': 4e-5,
              'warmup_ratio': 0.10,
              'train_batch_size': 16,
              'eval_batch_size': 16, 
              #'evaluate_during_training': False,
              #'evaluate_during_training_steps': 2000,    
              'do_lower_case': True,
              'silent':True,
              'reprocess_input_data': True, 
              'overwrite_output_dir': False,
              'output_dir': '%s_outputs/'%(data_type),
              'best_model_dir': "%s_outputs/best_model"%(data_type),
              'cache_dir': "%s_cache_dir/"%(data_type)})
    else:
        raise NotImplementedError
    
    if data_type == "sst":
        print ("Work on different datasets")
        train_df, test_df, ood_df = load_sst_dataset()
    elif data_type == "ROSTD":
        train_df, test_df, ood_df = load_ROSTD_dataset()
    else:
        print ("Work on different intents")
        ood_df = load_dataset('clinc150_oos_test', data_type=data_type) 
        train_df = load_dataset('clinc150_train', data_type=data_type)
        test_df  = load_dataset('clinc150_test', data_type=data_type)

    print ("train", len(train_df), "test", len(test_df), "ood", len(ood_df))
    for use_cls in [False, True]:
        if use_cls:
            print ("---------- Use [CLS] token to represent sequence ----------")
        else:
            print ("---------- Use AVG embebeddings to represent sequence ----------")
        mean_list, precision_list = model.sample_X_estimator(train_df['text'].values.tolist(), use_cls)
        
        if distance == "l2":   #  baseline of EDF
            test_mah_vanlia = model.get_alternative_distance_score(test_df['text'].values.tolist(), mean_list, use_cls)[:, 1:]
            ood_mah_vanlia = model.get_alternative_distance_score(ood_df['text'].values.tolist(), mean_list, use_cls)[:, 1:]
            train_mah_vanlia = model.get_alternative_distance_score(train_df['text'].values.tolist(), mean_list, use_cls)[:, 1:]
        else:    # MDF
            test_mah_vanlia = model.get_unsup_Mah_score(test_df['text'].values.tolist(), mean_list, precision_list, use_cls)[:, 1:]
            ood_mah_vanlia = model.get_unsup_Mah_score(ood_df['text'].values.tolist(), mean_list, precision_list, use_cls)[:, 1:]
            train_mah_vanlia = model.get_unsup_Mah_score(train_df['text'].values.tolist(), mean_list, precision_list, use_cls)[:, 1:]

        ood_labels = np.ones(shape=(ood_mah_vanlia.shape[0], ))
        test_labels = np.zeros(shape=(test_mah_vanlia.shape[0], ))
        
        test_mah_scores = test_mah_vanlia 
        ood_mah_scores = ood_mah_vanlia 
        train_mah_scores = train_mah_vanlia
        
        candidate_list = [1e-9, 1e-7, 1e-5, 1e-3, 0.01, 0.1, 0.2, 0.5]
        # candidate_list = [1e-9, 1e-7, 1e-5, 1e-3, 0.01, 0.1, 0.5, 0.7, 0.9]
        np.random.shuffle(test_mah_scores)
        np.random.shuffle(ood_mah_scores)
        best_ours_results = None
        best_ours_AUROC = 0.0
        # for k in ['poly', 'linear']:
        for k in ['linear']:
            for nuu in candidate_list:
                print ("running ---:", "kernel:", k, "nuu:", nuu)
                c_lr = svm.OneClassSVM(nu=nuu, kernel=k, degree=2)
                c_lr.fit(train_mah_scores)
                test_scores = c_lr.score_samples(test_mah_scores)
                ood_scores = c_lr.score_samples(ood_mah_scores)
                X_scores = np.concatenate((ood_scores, test_scores))
                Y_test = np.concatenate((ood_labels, test_labels))
                            
                results = detection_performance(X_scores, Y_test, 'mah_logs', tag='TMP')
                neg_resuls = detection_performance(-X_scores, Y_test, 'feats_logs', tag='TMP')
                if sum(results["TMP"].values()) < sum(neg_resuls["TMP"].values()):
                    results = neg_resuls
            
                if results['TMP']['AUROC'] > best_ours_AUROC:
                    best_ours_AUROC = results['TMP']['AUROC']
                    ans = compute_all_scores(test_scores, ood_scores, "./")
                    best_ours_results = results
                    best_hypers = '{}-{}'.format(k, nuu)
                    # save data for plotting 
                    d = {"X_scores": X_scores, "Y_test": Y_test, "Features": np.concatenate((test_mah_scores, ood_mah_scores))}
        mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*best_ours_results['TMP']['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*best_ours_results['TMP']['DTACC']), end='')
        print(' {val:6.2f}'.format(val=100.*best_ours_results['TMP']['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*best_ours_results['TMP']['AUOUT']), end='')
        print("best hyper %s"%(best_hypers)) 
        print ("saving data for plotting")
        with open("./outputs/{}_{}_{}.pkl".format(data_type, args.model_class, load_path.split("/")[-1]), "wb") as f:
            pickle.dump(d, f)
        print('-------------------------------')


def MSP_OOD_detect(args):
    data_type = args.data_type.strip()
    print("data type %s"%( data_type))
    if args.load_path:
        load_path = args.load_path
    else:
        load_path = 'bert-base-uncased' if args.model_class == 'bert' else 'roberta-base'
    if args.model_class == 'bert' or args.model_class == 'roberta':
        assert load_path is not None
        model = ClassificationModel(
        args.model_class, 
        load_path, 
        num_labels=2,
        use_cuda=True,
        cuda_device=int(args.gpu_id),
        args={'num_train_epochs': 0,
              'fp16':False,
              'n_gpu':int(args.n_gpu),
              'learning_rate': 4e-5,
              'warmup_ratio': 0.10,
              'train_batch_size': 16,
              'eval_batch_size': 16, 
              #'evaluate_during_training': False,
              #'evaluate_during_training_steps': 2000,    
              'do_lower_case': True,
              'silent':True,
              'reprocess_input_data': True, 
              'overwrite_output_dir': False,
              'output_dir': '%s_outputs/'%(data_type),
              'best_model_dir': "%s_outputs/best_model"%(data_type),
              'cache_dir': "%s_cache_dir/"%(data_type)})
    else:
        raise NotImplementedError
        
    if data_type == "sst":
        print ("Work on different datasets")
        train_df, test_df, ood_df = load_sst_dataset()
    else:
        print ("Work on different intents")
        ood_df = load_dataset('clinc150_oos_test', data_type=data_type) 
        train_df = load_dataset('clinc150_train', data_type=data_type)
        test_df  = load_dataset('clinc150_test', data_type=data_type)
        test_preds, test_outputs = model.predict(test_df['text'].values.tolist())
    
    ood_preds, ood_outputs = model.predict(ood_df['text'].values.tolist())
    T_list = [1, 2, 3, 4, 5, 10, 20, 100, 1000, 10000, 100000]  # sweep the best temepurature 
    
    best_MSP_AUC = - np.inf
    from scipy.special import softmax 
    for temperature in T_list:
        test_soft_scores = softmax(test_outputs / float(temperature), axis=1)
        ood_soft_scores = softmax(ood_outputs / float(temperature), axis=1)

        test_max_scores = np.max(test_soft_scores, axis=1)
        ood_max_scores = np.max(ood_soft_scores, axis=1)

        out_lables = np.ones_like(ood_max_scores)
        in_labels = np.zeros_like(test_max_scores)
        all_labels = np.concatenate([out_lables, in_labels])
        all_scores = np.concatenate([-ood_max_scores, -test_max_scores])

        # neg results is used to solve a problem detection_performance 
        results = detection_performance(all_scores, all_labels, 'baseline_logs', tag='msp')
        neg_results = detection_performance(-all_scores, all_labels, 'baseline_logs', tag='msp')
        if sum(results["msp"].values()) < sum(neg_results["msp"].values()):
            results = neg_results
            all_scores = - all_scores

        if temperature == 1:
            baseline_results = results
        
        if results['msp']['AUROC'] > best_MSP_AUC:
            best_MSP_AUC = results['msp']['AUROC']
            best_MSP_results = results
            best_MSP_hypers = temperature
            d = {"X_scores": all_scores, "Y_test": all_labels}

    mtypes = ['AUROC', 'DTACC', 'AUIN', 'AUOUT']
    print("Best ODIN T= %s"%(best_MSP_hypers))
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100.*best_MSP_results['msp']['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*best_MSP_results['msp']['DTACC']), end='')
    print(' {val:6.2f}'.format(val=100.*best_MSP_results['msp']['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*best_MSP_results['msp']['AUOUT']), end='')
    print('saving results for MSP')
    with open("./outputs/MSP_{}_{}.pkl".format(data_type, args.model_class), "wb") as f_:
        pickle.dump(d, f_)


def main(args):
    global seed
    seed = args.seed
    method = args.method
    # set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(args.gpu_id)
    if method == 'MDF':
        MDF_OOD_detect(args)   # our method 
    elif method == 'single_layer_bert':  # check performance of a single layer of BERT 
        single_layer_OOD_detect(args)
    elif method == "MSP":
        MSP_OOD_detect(args)
    elif method == "tf-idf":
        frequency_OOD_detect(args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Bert Model OOD Detection")
    parser.add_argument('--method', default='MDF', choices=['MDF', 'single_layer_bert', 'tf-idf', 'MSP'])
    parser.add_argument('--model_class', default='bert', choices=['bert', 'roberta'])
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--n_rep', default=1, type=int)
    parser.add_argument('--n_gpu', default=1, type=int)
    parser.add_argument('--seed', default=171, type=int)
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--data_type', default='ROSTD', choices = ['clinic', 'sst', "ROSTD"])

    args = parser.parse_args()
    main(args)
