import json

from dataloader.grailqa_json_loader import GrailQAJsonLoader
from utils.domain_dict import fb_domain_dict
from utils.cached_enumeration import date_relations, numerical_relations
from utils.config import grailqa_train_path, grailqa_dev_path, grailqa_test_path


def init_statistics_dicts():
    predicate_count = dict()
    type_count = dict()

    schema_set = set()
    for domain in fb_domain_dict:
        for schema in fb_domain_dict[domain]:
            schema_set.add(schema)

    for schema in schema_set:
        pos = schema.rfind('.')
        if schema[:pos] in schema_set:
            predicate_count[schema] = 0
        else:
            type_count[schema] = 0
    return predicate_count, type_count


def get_grailqa_all_predicates_and_types():
    type_set = set()
    predicate_set = set()

    schema_set = set()
    for domain in fb_domain_dict:  # domain -> predicate / type
        for schema in fb_domain_dict[domain]:
            schema_set.add(schema)

    for schema in schema_set:
        pos = schema.rfind('.')
        if schema[:pos] in schema_set:  # the type prefix is already in this set
            predicate_set.add(schema)
        else:
            type_set.add(schema)
    return predicate_set, type_set


def get_dataset_statistics(dataloader: GrailQAJsonLoader, predicate_count=None, type_count=None, domain_count=None):
    if predicate_count is None:
        predicate_count, type_count = init_statistics_dicts()
    if domain_count is None:
        domain_count = dict()

    for idx in range(0, dataloader.len):  # for each question
        predicates = dataloader.get_golden_relation_by_idx(idx)
        for predicate in predicates:
            predicate_count[predicate] += 1
            for t in type_count:
                if t in predicate:
                    type_count[t] += 1
            for domain in fb_domain_dict:
                if domain in predicate:
                    domain_count[domain] = domain_count.get(domain, 0) + 1
    return predicate_count, type_count, domain_count


if __name__ == '__main__':
    grail_train_data = GrailQAJsonLoader(grailqa_train_path)
    grail_dev_data = GrailQAJsonLoader(grailqa_dev_path)
    grail_test_data = GrailQAJsonLoader(grailqa_test_path)

    train_stat = get_dataset_statistics(grail_train_data)
    # print(train_stat[0])
    # print(len(train_stat[0]))
    # print()
    dev_stat = get_dataset_statistics(grail_dev_data)
    # print(dev_stat[0])
    # print(len(dev_stat[0]))

    num_entity_pair = dict()
    with open('../dataset/GrailQA/relation_num_entity_pair.tsv') as f:
        for line in f:
            split = line.strip('\n').split('\t')
            num_entity_pair[split[1]] = int(split[2])
    stat = []
    zero_shot_no_text = 0
    zero_shot_avg_entity_pair = 0
    for schema in set(train_stat[0].keys()).union(dev_stat[0].keys()):
        d = {"relation": schema, "#train": train_stat[0].get(schema, 0), "#dev": dev_stat[0].get(schema, 0)}
        d["seen in train"] = True if d["#train"] != 0 else False
        d["none for train and dev"] = True if d["#train"] == 0 and d["#dev"] == 0 else False
        d["only in train"] = True if d["#train"] != 0 and d["#dev"] == 0 else False
        d["i.i.d / compositional"] = True if d["#train"] != 0 and d["#dev"] != 0 else False
        d["zero-shot"] = True if d["#train"] == 0 and d["#dev"] != 0 else False
        d["date_relation"] = True if d["relation"] in date_relations else False
        d["numerical_relation"] = True if d["relation"] in numerical_relations else False
        d["#entity_pair"] = num_entity_pair.get(schema, 0)
        stat.append(d)
        if d["zero-shot"]:
            if d['#entity_pair'] == 0:
                zero_shot_no_text += 1
            zero_shot_avg_entity_pair += d['#entity_pair']
    print(json.dumps(stat))
    print(zero_shot_no_text)
    print(zero_shot_avg_entity_pair)
    # for domain in dev_stat[2]:
    #     if domain not in train_stat[2]:
    #         print(domain)
