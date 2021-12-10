# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from datetime import datetime

month2num = {"jan.": '1',
             "feb.": '2',
             "mar.": '3',
             "apr.": '4',
             "may": '5',
             "june": '6',
             "july": '7',
             "aug.": '8',
             "sept.": '9',
             "oct.": '10',
             "nov.": '11',
             "dec.": '12'}
literal_prefix = "http://www.w3.org/2001/"
date_regex = re.compile(r"(\d+) the (\d+)(.+) , (\d+)")
year_regex = re.compile(r"[1-2][0-9]{3}")

def normalize_date(literal):
    literal = literal.replace("/", "-")
    try:
        literal_toks = literal.split()
        trans_literal = " ".join([month2num[tok] if tok in month2num else tok for tok in literal_toks])
        match_obj = date_regex.match(trans_literal)
        if match_obj:
            month = match_obj.group(1)
            if len(month) == 1:
                month = "0{}".format(month)
            day = match_obj.group(2)
            if len(day) == 1:
                day = "0{}".format(day)
            year = match_obj.group(4)
            literal = "{}-{}-{}".format(year, month, day)
        return literal
    except:
        return literal


def pack_one_literal_node(nid, literal, literal_type, start, end):
    if literal_type == "XMLSchema#gYear" or literal_type == "XMLSchema#date" \
        or literal_type == "XMLSchema#gYearMonth" or literal_type == "XMLSchema#dateTime":
        cls = "type.datetime"
        id = "{}^^{}{}".format(normalize_date(literal), literal_prefix, literal_type)
    elif literal_type == "type.int":
        cls = literal_type
        id = "{}^^{}{}".format(literal, literal_prefix, "XMLSchema#integer")
    elif literal_type == "type.float":
        cls = literal_type
        id = "{}^^{}{}".format(literal, literal_prefix, "XMLSchema#float")
    else:
        # type.boolean
        cls = literal_type
        id = "{}^^{}{}".format(literal, literal_prefix, "XMLSchema#integer")

    return {"nid": nid,
            "node_type": "literal",
            "id": id,
            "class": cls,
            "friendly_name": literal,
            "question_node": 0,
            "offset": (start, end),
            "function": 'none'}


def gen_all_literal_nodes(el_output, world="webQSP"):
    if world == "WebQSP":
        literals = []
        tokens = el_output["tokens"]
        for i, tok in enumerate(tokens):
            if year_regex.match(tok):
                literals.append((tok, "XMLSchema#gYear", i, i+1))
    else:
        mentions = el_output["mentions"]
        tokens = el_output["tokens"]
        literals = []
        for x in mentions:
            mention = " ".join(tokens[x["start"]:x["end"]])
            mention_type = x["type"]
            if mention_type == "XMLSchema#gYear" or mention_type == "XMLSchema#date" \
                    or mention_type == "XMLSchema#gYearMonth" or mention_type == "XMLSchema#dateTime" \
                    or mention_type == "type.float" or mention_type == "type.int" or mention_type == "type.boolean":
                literals.append((mention, mention_type, x["start"], x["end"]))

    node_id = 0
    literal_nodes = []
    for x in literals:
        literal_nodes.append(pack_one_literal_node(node_id, x[0], x[1], x[2], x[3]))
        node_id += 1
    return literal_nodes, node_id


class Entity(object):
    def __init__(self, ent_id, offset, score):
        self.ent_id = ent_id
        self.offset = offset
        self.score = score

def get_prior_el_topk(sent, topk=5):
    entity_list = []
    for i, x in enumerate(sent["topk_entities"]):
        for j in range(min(len(x), topk)):
            if sent["topk_scores"][i][j] < 0.01:
                continue
            entity_list.append(Entity(x[j], sent["boundary"][i], sent["topk_scores"][i][j]))
    return entity_list


class Logger(object):
    def __init__(self, log_fn):
        self.log_fn = log_fn

    def log(self, text):
        with open(self.log_fn, encoding="utf-8", mode="a") as fp:
            fp.write("{}\t{}\n".format(datetime.now(), text))
