# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import inflection as inflection
import urllib.request

# RDF
rdf_type_uri = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'
rdf_label_uri = 'http://www.w3.org/2000/01/rdf-schema#label'
rdf_comment_uri = 'http://www.w3.org/2000/01/rdf-schema#comment'
rdf_property_uri = 'http://www.w3.org/2000/01/rdf-schema#Property'
rdf_class_uri = 'http://www.w3.org/2000/01/rdf-schema#Class'
rdf_domain_uri = 'http://www.w3.org/2000/01/rdf-schema#domain'
rdf_range_uri = 'http://www.w3.org/2000/01/rdf-schema#range'
# XMLSchema
schema_integer = 'http://www.w3.org/2001/XMLSchema#integer'
schema_float = 'http://www.w3.org/2001/XMLSchema#float'
# Freebase
ns_prefix = 'http://rdf.freebase.com/ns/'
key_prefix = 'http://rdf.freebase.com/key/'
freebase_type_uri = 'http://rdf.freebase.com/ns/type.object.type'
freebase_instance_uri = 'http://rdf.freebase.com/ns/type.type.instance'
# DBpedia
dbr_prefix = 'http://dbpedia.org/resource/'
dbp_prefix = 'http://dbpedia.org/property/'
dbo_prefix = 'http://dbpedia.org/ontology/'
dbc_prefix = 'http://dbpedia.org/property/class/'
# Wikidata
wd_prefix = 'http://www.wikidata.org/entity/'
prop_prefix = 'http://www.wikidata.org/prop/'
wdt_prefix = 'http://www.wikidata.org/prop/direct/'
ps_prefix = 'http://www.wikidata.org/prop/statement/'
pq_prefix = 'http://www.wikidata.org/prop/qualifier/'


def remove_ns(uri: str) -> str:
    assert uri is not None
    if isinstance(uri, dict):
        uri = uri['value']
    if uri.startswith(ns_prefix):
        return uri[27:]
    if uri.startswith(key_prefix):
        return uri[28:]
    return uri


def remove_freebase_prefix(uri: str) -> str:
    assert uri is not None
    if uri.startswith('www.freebase.com/'):
        return uri[17:].replace('/', '.')
    return uri.replace('/', '.')


def remove_prefix(uri: str) -> str:
    return uri.split('/')[-1]


def remove_dbpedia_prefix(uri: str) -> str:
    if uri.startswith(dbp_prefix) or uri.startswith(dbo_prefix) or uri.startswith(dbr_prefix):
        return uri[28:]
    return uri


def remove_list_ns(uri_list: list) -> list:
    for i in range(0, len(uri_list)):
        uri_list[i] = remove_ns(uri_list[i])
    return uri_list


def remove_set_ns(uri_set: set) -> set:
    res = set()
    for uri in uri_set:
        res.add(remove_ns(uri))
    return res


def ns(label: str) -> str:
    assert label is not None
    if isinstance(label, dict):
        label = label['value']
    if label.startswith(ns_prefix) or label.startswith('?'):  # don't add this prefix
        return label
    return ns_prefix + label


def ns_list(label_list: list) -> list:
    res = []
    for label in label_list:
        res.append(ns(label))
    return res


def brackets(s: str) -> str:
    if s.startswith("http"):
        return "<" + s + ">"
    return s


def brackets_triple(s, p, o) -> str:
    return brackets(s) + ' ' + brackets(p) + ' ' + brackets(o)


def substr_before_brackets(uri: str) -> str:
    uri = remove_dbpedia_prefix(uri)
    bracket_idx = uri.find('(')
    if bracket_idx != -1:
        uri = uri[: bracket_idx]
    uri = uri.replace('_', ' ').strip(' ').lower()
    return uri


def word_list_before_brackets(uri: str) -> list:
    return substr_before_brackets(uri).split(' ')


def remove_brackets(s: str) -> str:
    if s.startswith('<') and s.endswith('>'):
        s = s[1: -1]
    return s


def add_question_mark(var: str) -> str:
    if var.startswith('?'):
        return var
    return '?' + var


def add_quotation_mark(literal: str) -> str:
    if literal.startswith('"') and literal.endswith('"'):
        return literal
    return '"' + literal + '"'


def remove_question_mark(var: str) -> str:
    if var.startswith('?'):
        return var[1:]
    return var


def remove_quotation_mark(literal: str) -> str:
    if literal.startswith('"') and literal.endswith('"'):
        return literal[1:-1]
    return literal


def is_mid_gid(s: str) -> bool:
    if s is None:
        return False
    if s.startswith('m.') or s.startswith('g.') or 'ns/m.' in s or 'ns/g.' in s:
        return True
    return False


def has_mid_gid(s: str) -> bool:
    if 'ns:m.' in s or 'ns:g.' in s or 'ns/m.' in s or 'ns/g.' in s or ' m.' in s or ' g.' in s or s.startswith(
            'm.') or s.startswith('g.'):
        return True
    return False


def is_dbp_dbo(uri: str) -> bool:
    if uri.startswith(dbp_prefix) or len(re.findall('http://dbpedia.org/ontology/[a-z]', uri)):
        return True
    return False


def is_dbr_dbo(uri: str) -> bool:
    if uri.startswith(dbr_prefix) or len(re.findall('http://dbpedia.org/ontology/[A-Z]', uri)):
        return True
    return False


def is_dbr(uri: str) -> bool:
    if uri.startswith(dbr_prefix):
        return True
    return False


def get_dbp_dbo_relation_list_in_sparql(sparql: str) -> list:
    res = []
    dbp_list = re.findall('<http://dbpedia.org/property/.*?>', sparql)
    for dbp in set(dbp_list):
        res.append(remove_brackets(dbp))
    dbo_list = re.findall('<http://dbpedia.org/ontology/[a-z].*?>', sparql)
    for dbo in set(dbo_list):
        res.append(remove_brackets(dbo))
    return res


def get_dbr_dbo_list_in_sparql(sparql: str) -> list:
    res = []
    res += get_dbr_entity_list_in_sparql(sparql)
    res += get_dbo_class_list_in_sparql(sparql)
    return set(res)


def get_dbr_entity_list_in_sparql(sparql: str) -> list:
    res = []
    dbr_list = re.findall('<http://dbpedia.org/resource/.*?>', sparql)
    for dbr in set(dbr_list):
        res.append(remove_brackets(dbr))
    return res


def get_dbo_class_list_in_sparql(sparql: str) -> list:
    res = []
    dbo_list = re.findall('<http://dbpedia.org/ontology/[A-Z].*?>', sparql)
    for dbo in set(dbo_list):
        res.append(remove_brackets(dbo))
    return res


def relation_label_to_words(label: str) -> str:
    label = remove_ns(label)
    return label.replace('.', ' ').replace('_', ' ')


def relation_label_to_domain_and_property(label: str) -> str:
    label = remove_ns(label)
    return label[label.find('.') + 1:].replace('.', ' ').replace('_', ' ')


def schema_domain(label: str) -> str:
    label = remove_ns(label)
    return label[:label.find('.')]


def schema_set_domain(labels) -> set:
    if labels is None:
        return set()
    return set([schema_domain(label) for label in labels])


def relation_label_to_property(uri: str) -> str:
    uri = remove_ns(uri)
    return uri.split('.')[-1].replace('_', ' ')


def relation_label_list_to_words(uri_list: list) -> list:
    res = []
    for uri in uri_list:
        res.append(relation_label_to_words(uri))
    return res


def relation_set_filter(s: set, relation_white_list=None) -> set:
    black_list = [freebase_type_uri, rdf_type_uri, 'wikipedia', 'type.type.instance', 'webpage',
                  'base.ontologies.ontology_instance.equivalent_instances',
                  'base.ontologies.ontology_instance_mapping.freebase_topic', 'kg.object_profile.prominent_type',
                  'authority.facebook', 'youtube_channel']
    res = set()
    for uri in s:
        black = False
        for b in black_list:
            if b in uri:
                black = True
                break
        if black is False and relation_white_list is not None:
            if remove_ns(uri) not in relation_white_list:
                black = True
        if not black:
            res.add(uri)
    return res


def freebase_rdf_type_list_to_str(rdf_type_list):
    res = ''
    for rdf_type in rdf_type_list:
        res += rdf_type.split('.')[-1].replace('_', ', ') + '; '
    return res[:-2]


def dbpedia_uri_to_words(uri: str) -> str:
    # uri = uri.replace(dbo_prefix, 'ontology, ')
    uri = uri.replace(dbo_prefix, '')
    # uri = uri.replace(dbp_prefix, 'property, ')
    uri = uri.replace(dbp_prefix, '')
    uri = uri.replace(dbr_prefix, '')
    uri = inflection.underscore(uri).replace('_', ' ').strip(' ')
    # CamelCase to snake_case, then replace
    return uri


def get_dbr_labels(dbr):
    label = dbpedia_uri_to_words(dbr)
    labels = [label]
    bracket_idx = label.find('(')
    if bracket_idx != -1:
        label_wo_bracket = label[: bracket_idx].strip(' ')
        labels.append(label_wo_bracket)
    return labels


def dbp_dbo_conversion(uri: str) -> str:
    if uri.startswith(dbp_prefix):
        return uri.replace(dbp_prefix, dbo_prefix)
    if uri.startswith(dbo_prefix):
        return uri.replace(dbo_prefix, dbp_prefix)
    return uri


def get_html(uri: str) -> str:
    fp = urllib.request.urlopen(uri)
    mybytes = fp.read()
    res = mybytes.decode("utf8")
    fp.close()
    return res
