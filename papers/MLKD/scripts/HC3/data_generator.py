# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import datasets
import random
import os
os.makedirs("OOD_dataset/HC3", exist_ok=True)


d = datasets.load_dataset("Hello-SimpleAI/HC3", "all")
iid, ood = [], []
iid_ids = []
idx = 0

for ii in d["train"]:
    a = [i.replace("\r", "").replace("\n", "") for i in ii["chatgpt_answers"]]
    a = [i for i in a if i.strip()]
    iid.extend(a)
    iid_ids.extend(list(range(idx, idx + len(a))))
    idx += len(a)
    a = [i.replace("\r", "").replace("\n", "") for i in ii["human_answers"]]
    a = [i for i in a if i.strip()]
    ood.extend(a)

random.seed(42)
random.shuffle(iid)
random.shuffle(ood)


with open("OOD_dataset/HC3/hc3_train.txt", "w") as f:
    f.write("\n".join(iid[:len(iid)//2]))
with open("OOD_dataset/HC3/hc3_test.txt", "w") as f:
    f.write("\n".join(iid[len(iid)//2:]))
with open("OOD_dataset/HC3/hc3_ood.txt", "w") as f:
    f.write("\n".join(ood))


random.seed(42)
random.shuffle(iid_ids)
train_id = set(iid_ids[:len(iid_ids) // 2])

iid, ood = [], []
iid_ids = []
idx = 0

for ii in d["train"]:
    a = [i.replace("\r", "").replace("\n", "") for i in ii["chatgpt_answers"]]
    a = [i for i in a if i.strip()]
    # idx += len(a)
    b = [i.replace("\r", "").replace("\n", "") for i in ii["human_answers"]]
    b = [i for i in b if i.strip()]
    for (k, i), j in zip(enumerate(a), b):
        if idx + k not in train_id:
            iid.append(i)
            ood.append(j)
    idx += len(a)

with open("OOD_dataset/HC3/hc3_pair_test.txt", "w") as f:
    f.write("\n".join(iid))
with open("OOD_dataset/HC3/hc3_pair_ood.txt", "w") as f:
    f.write("\n".join(ood))