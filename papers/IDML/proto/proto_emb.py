import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import json
import argparse
from encoders.bert_cls_encoder import BERTEncoder
from encoders.sbert_encoder import SentEncoder
from encoders.bert_mean_encoder import BERTMeanEncoder
from encoders.para_encoder import ParaEncoder
from utils.dataloader import FewShotDataLoader
from utils.train_utils import set_seeds, euclidean_dist, cosine_similarity
import collections
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import warnings
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_label_file(label_fn):
    class2name_mp = {}
    replace_pair = [("lightdim", "light dim"), ("lightchange", "light change"), ("lightup", "light up"),
                    ("commandstop", "command stop"), ("lighton", "light on"), ("dontcare", "don't care"),
                    ("lightoff", "light off"), ("querycontact", "query contact"), ("addcontact", "add contact"),
                    ("sendemail", "send email"), ("createoradd", "create or add"), ("qa", "what")]

    with open(label_fn, mode="r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip("\n")
            tmp = line.replace("_", " ").replace("/", " ")
            for x, y in replace_pair:
                tmp = tmp.replace(x, y)
            class2name_mp[line] = tmp
    return class2name_mp

class ProtoNet(nn.Module):
    def __init__(self, encoder, metric="euclidean", label_fn=None):
        super(ProtoNet, self).__init__()
        self.encoder = encoder
        self.metric = metric
        assert self.metric in ('euclidean', 'cosine')
        self.class2name_mp = load_label_file(label_fn) if label_fn else None


    def eval_with_log(self, sample, classes, pooling):
        xs = sample['xs']  # support
        xq = sample['xq']  # query
        supports = [item["sentence"] for xs_ in xs for item in xs_]
        queries = [item["sentence"] for xq_ in xq for item in xq_]
        gold_labels = [classes[i] for i in range(len(xq)) for j in xq[i]]
        sp_gold_labels = [classes[i] for i in range(len(xs)) for j in xs[i]]
        loss, loss_dict = self.loss(sample, classes, pooling)
        logs = {"support": [], "query": [], "accuracy": loss_dict["metrics"]["acc"]}
        for i in range(len(supports)):
            logs["support"].append({"words": supports[i], "gold": sp_gold_labels[i]})
        for i in range(len(queries)):
            logs["query"].append({"words": queries[i], "gold": gold_labels[i], "pred": classes[loss_dict["pred"][i]]})
        return logs

    def loss(self, sample, classes=None, pooling='avg'):
        xs = sample['xs']  # support
        xq = sample['xq']  # query

        n_class = len(xs)
        assert len(xq) == n_class
        n_support = len(xs[0])
        n_query = len(xq[0])

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False).to(device)

        # When not using augmentations
        supports = [item["sentence"] for xs_ in xs for item in xs_]
        queries = [item["sentence"] for xq_ in xq for item in xq_]

        # Encode
        x = supports + queries
        z = self.encoder.embed_sentences(x)
        z_dim = z.size(-1)

        # Dispatch
        z_support = z[:len(supports)].view(n_class, n_support, z_dim)
        if self.class2name_mp:
            class_names = [self.class2name_mp[classes[i]] for i in range(len(xs))]
            z_class = self.encoder.embed_sentences(class_names).view(n_class, 1, z_dim)
            z_support = torch.cat([z_support, z_class], dim=1)

        z_query = z[len(supports):len(supports) + len(queries)]
        if pooling == 'avg':
            z_support = z_support.mean(dim=[1])
        else:
            assert pooling == 'nn'
            z_support = z_support.view(-1, z_dim)
        if self.metric == "euclidean":
            supervised_dists = euclidean_dist(z_query, z_support)
        elif self.metric == "cosine":
            supervised_dists = (-cosine_similarity(z_query, z_support) + 1) * 5
        else:
            raise NotImplementedError

        if pooling == 'nn':
            supervised_dists = supervised_dists.view(len(queries), n_class, -1)
            # print(supervised_dists.shape)
            supervised_dists = torch.min(supervised_dists, 2)[0]
        # Supervised loss
        supervised_loss = CrossEntropyLoss()(-supervised_dists, target_inds.reshape(-1))
        _, y_hat_supervised = (-supervised_dists).max(1)
        acc_val_supervised = torch.eq(y_hat_supervised, target_inds.reshape(-1)).float().mean()

        return supervised_loss, {
            "metrics": {
                "acc": acc_val_supervised.item(),
                "loss": supervised_loss.item(),
            },
            "dists": supervised_dists,
            "target": target_inds.cpu().tolist(),
            "pred": y_hat_supervised.cpu().tolist()
        }

    def test_step(self,
                  data_loader: FewShotDataLoader,
                  n_support: int,
                  n_query: int,
                  n_classes: int,
                  n_episodes: int = 1000):
        metrics = collections.defaultdict(list)

        self.eval()
        for i in range(n_episodes):
            episode, classes = data_loader.create_episode(
                n_support=n_support,
                n_query=n_query,
                n_classes=n_classes,
            )
            with torch.no_grad():
                loss, loss_dict = self.loss(episode)

            for k, v in loss_dict["metrics"].items():
                metrics[k].append(v)

        return {
            key: np.mean(value) for key, value in metrics.items()
        }

def test_proto(
        encoder: str,
        model_name_or_path: str,
        n_support: int,
        n_query: int,
        n_classes: int,
        test_path: str = None,
        output_path: str = './output',
        n_test_episodes: int = 600,
        metric: str = "euclidean",
        label_fn: str = None,
        load_ckpt: bool = False,
        pooling: str = 'avg'
):

    logs = []
    # Load model
    if encoder == "bert":
        bert = BERTEncoder(model_name_or_path).to(device)
    elif encoder == "bertmean":
        bert = BERTMeanEncoder(model_name_or_path).to(device)
    elif encoder == "sentbert":
        bert = SentEncoder(model_name_or_path).to(device)
    elif encoder == "para":
        bert = ParaEncoder(load_file=os.path.join(model_name_or_path, "model.para.lc.100.pt"),
                           sp_model=os.path.join(model_name_or_path, "paranmt.model")).to(device)
    else:
        raise ValueError("encoder name unk")
        
    protonet = ProtoNet(encoder=bert, metric=metric, label_fn=label_fn)
    if load_ckpt:
        protonet.load_state_dict(torch.load(os.path.join(output_path, "encoder.pkl")))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Load data
    valid_data_loader = FewShotDataLoader(test_path)
    logger.info(f"valid labels: {valid_data_loader.data_dict.keys()}")

    protonet.eval()
    for i in range(n_test_episodes):
        episode, classes = valid_data_loader.create_episode(
            n_support=n_support,
            n_query=n_query,
            n_classes=n_classes,
        )
        with torch.no_grad():
            logs.append(protonet.eval_with_log(episode, classes, pooling))
    acc = []
    with open(os.path.join(output_path, "logs.json"), mode="w", encoding="utf-8") as fp:
        for line in logs:
            fp.write(json.dumps(line) + "\n")
            fp.flush()
            acc.append(line["accuracy"])
    avg_acc = np.mean(acc)
    with open(os.path.join(output_path, 'metrics.json'), "w") as file:
        json.dump({"accuracy": avg_acc}, file, ensure_ascii=False)
    return avg_acc

def str2bool(arg):
    if arg.lower() == "true":
        return True
    return False

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=str, default=None, help="Path to testing data")
    parser.add_argument("--output-path", type=str, default=None, required=True)

    parser.add_argument("--encoder", type=str, default="bert")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Transformer model to use")
    parser.add_argument("--load_ckpt", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, default=5, help="Number of support points for each class")
    parser.add_argument("--n-query", type=int, default=5, help="Number of query points for each class")
    parser.add_argument("--n-classes", type=int, default=5, help="Number of classes per episode")
    parser.add_argument("--n-test-episodes", type=int, default=600, help="Number of episodes during evaluation (valid, test)")

    #currently it is only for test...
    parser.add_argument("--label_fn", type=str, default=None)

    # Metric to use in proto distance calculation
    parser.add_argument("--metric", type=str, default="euclidean", help="Metric to use", choices=("euclidean", "cosine"))
    parser.add_argument("--pooling", type=str, default="avg", help="Metric to use", choices=("avg", "nn"))

    args = parser.parse_args()
    return args

def read_results(output_dir):
    for model_name in ["paraphrase-distilroberta-base-v2-euc-wotrain-label",
                        "nli-roberta-base-v2-euc-wotrain-label",
                        "simcse-nli-euc-wotrain-label",
                        "declutr-base-euc-wotrain-label",
                        "sp-paraphrase-cos-wotrain-label"
                        ]:
        for K in [1, 5]:
            acc_list = []
            for split in [1, 2, 3, 4, 5]:
                res_fn = os.path.join(output_dir, f"0{split}/proto-5way{K}shot-{model_name}/metrics.json")
                with open(res_fn, mode="r", encoding="utf-8") as fp:
                    acc = json.load(fp)["accuracy"]
                    acc_list.append(acc)
            mu = np.mean(acc_list) * 100
            std = np.std(acc_list) * 100
            print("model {}, {} shot : {} +- {}".format(model_name, K, mu, std))
    return

def main(args):
    # Set random seed
    set_seeds(args.seed)
    avg_acc = test_proto(
        encoder=args.encoder,
        model_name_or_path=args.model_name_or_path,
        n_support=args.n_support,
        n_query=args.n_query,
        n_classes=args.n_classes,
        n_test_episodes=args.n_test_episodes,
        output_path=args.output_path,
        metric=args.metric,
        test_path=args.test_path,
        label_fn=args.label_fn,
        load_ckpt=args.load_ckpt,
        pooling=args.pooling,
    )
    #Save config
    with open(os.path.join(args.output_path, "config.json"), "w") as file:
        json.dump(vars(args), file, ensure_ascii=False)
    return avg_acc

if __name__ == '__main__':
    # read_results("./output/BANKING77")
    args = add_args()
    acc = main(args)
    print(acc)