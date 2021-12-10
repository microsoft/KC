# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import codecs
import numpy as np  
import matplotlib.pyplot as plt 

f=open('webQSPtest.json','r')
data_l=json.load(f)
mode="test"

f_cls_id=open('freebase_class_id_webqsp','r')
f_rel_id=open('freebase_relation_id_webqsp','r')
cls2id=json.load(f_cls_id)
rel2id=json.load(f_rel_id)
classes=[]
relations=[]


cnt=0
cnt_relation=0
cnt_class=0

relation_pres=[]
class_pres=[]
k_rel=200
k_cls=200

def from_str(s):
    return json.loads(s)


def from_file(path):
    try:
        with codecs.open(path, "r", "UTF-8") as f:
            return from_str(''.join(f.readlines()))
    except Exception:
        print("Absolute path: " + os.path.abspath(path))
        raise

#Extract predictions
class_pre_d=from_file("cls_predictions")
for c in class_pre_d:
    curr_pre=c["predictions"].split('\t')
    curr_pre_score=c["scores"].split('\t')
    tmp=[]
    for n in range(k_cls):
        tmp.append((curr_pre[n],curr_pre_score[n]))
        cnt_class+=1
    class_pres.append(tmp)

relation_pre_d=from_file("rel_predictions")
for r in relation_pre_d:
    curr_pre=r["predictions"].split('\t')
    curr_pre_score=r["scores"].split('\t')
    tmp=[]
    for n in range(k_rel):
        tmp.append((curr_pre[n],curr_pre_score[n]))
        cnt_relation+=1
    relation_pres.append(tmp)

#Pack predictions and gold ans
f_res=open('WebQSP%sRes_Rel%s_Cls%s.jsonl'%(mode,str(k_rel),str(k_cls)),'w')
f_gold=codecs.open("webqsp_gold_ans_%s"%mode,"w","utf-8")
gold_ans=[]
for d_l in data_l[:len(relation_pres)]:
    exp= {
            "qid": d_l["qid"],
            "classes":class_pres[cnt],
            "relations":relation_pres[cnt],
        }
    cnt+=1
    f_res.write(json.dumps(exp)+"\n")

    tmp_cls=[]
    tmp_rel=[]
    mention=d_l["question"]
   
    if len(d_l["kb_items"])==0:
        d_l["kb_items"]=[i.split(" ")[0] for i in d_l["sparql_query"].split("ns:")]
    
    for items in d_l["kb_items"]:
        if items in cls2id:
            tmp_cls.append(items)
        if items in rel2id:
            tmp_rel.append(items)

    golds={
        "relations":tmp_rel,
        "classes":tmp_cls
    }
    gold_ans.append(golds)

json.dump(gold_ans,f_gold)
f_gold.close()
f_res.close()

