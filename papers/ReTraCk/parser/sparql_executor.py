# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import time

from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

sparql = SPARQLWrapper("http://localhost:8890/sparql/")
prefix = "http://rdf.freebase.com/ns/"


def set_sparql_wrapper(uri):
    global sparql
    sparql = SPARQLWrapper(uri)


def exec_sparql(query):
    status_code = 200
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    except SPARQLExceptions.EndPointInternalError:
        status_code = 500
    except SPARQLExceptions.QueryBadFormed:
        status_code = 400
    except SPARQLExceptions.EndPointNotFound:
        status_code = 404
    except SPARQLExceptions.Unauthorized:
        status_code = 401
    except SPARQLExceptions.URITooLong:
        status_code = 414
    except Exception as e:
        print(e)
        status_code = -1
    pred_answer = []
    if status_code != 200:
        time.sleep(5)
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
        except:
            return pred_answer, status_code
    status_code = 200
    # print(query)
    for result in results["results"]["bindings"]:
        if 'x' in result:
            value = result["x"]["value"]
        elif 'callret-0' in result:
            value = result['callret-0']['value']
        elif 'value' in result:
            value = result['value']['value']
        else:
            raise Exception("UNKNOWN {}".format(result))
        if value.startswith(prefix):
            value = value[len(prefix):]
        value = value.replace("-08:00", '')
        pred_answer.append(value)
    # print(pred_answer)
    return pred_answer, status_code


def exec_demo_sparql(query):
    status_code = 200
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    except SPARQLExceptions.EndPointInternalError:
        status_code = 500
    except SPARQLExceptions.QueryBadFormed:
        status_code = 400
    except SPARQLExceptions.EndPointNotFound:
        status_code = 404
    except SPARQLExceptions.Unauthorized:
        status_code = 401
    except SPARQLExceptions.URITooLong:
        status_code = 414
    except:
        status_code = -1
    meta_info = {}
    if status_code != 200:
        time.sleep(5)
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
        except:
            return meta_info, status_code
    status_code = 200
    for result in results["results"]["bindings"]:
        meta_info[result["ent_id"]["value"][len(prefix):]] = {
            "ent_desc": result["ent_desc"]["value"] if "ent_desc" in result else "",
            "ent_name": result["ent_name"]["value"] if "ent_name" in result else "",
            "ent_type": result["ent_type"]["value"][len(prefix):] if "ent_type" in result else ""
        }
    return meta_info, status_code


def exec_schema_demo_sparql(query):
    status_code = 200
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    except SPARQLExceptions.EndPointInternalError:
        status_code = 500
    except SPARQLExceptions.QueryBadFormed:
        status_code = 400
    except SPARQLExceptions.EndPointNotFound:
        status_code = 404
    except SPARQLExceptions.Unauthorized:
        status_code = 401
    except SPARQLExceptions.URITooLong:
        status_code = 414
    except:
        status_code = -1
    meta_info = {}
    if status_code != 200:
        time.sleep(5)
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
        except:
            return meta_info, status_code
    status_code = 200
    for result in results["results"]["bindings"]:
        meta_info[result["schema_id"]["value"][1:].replace("/", ".")] = {
            "ent_id": result["ent_id"]["value"][len(prefix):],
            "ent_desc": result["ent_desc"]["value"] if "ent_desc" in result else "",
            "ent_name": result["ent_name"]["value"] if "ent_name" in result else "",
            "ent_type": result["ent_type"]["value"][len(prefix):] if "ent_type" in result else ""
        }
    return meta_info, status_code


def exec_anchor_relation_sparql(query, rel_name="in_rel"):
    status_code = 200
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    except SPARQLExceptions.EndPointInternalError:
        status_code = 500
    except SPARQLExceptions.QueryBadFormed:
        status_code = 400
    except SPARQLExceptions.EndPointNotFound:
        status_code = 404
    except SPARQLExceptions.Unauthorized:
        status_code = 401
    except SPARQLExceptions.URITooLong:
        status_code = 414
    except:
        status_code = -1
    meta_info = {}
    if status_code != 200:
        time.sleep(5)
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
        except:
            return meta_info, status_code
    status_code = 200
    for result in results["results"]["bindings"]:
        ent_id = result["ent_id"]["value"][len(prefix):]
        if ent_id not in meta_info:
            meta_info[ent_id] = []
        meta_info[ent_id].append(result[rel_name]["value"][len(prefix):])
    return meta_info, status_code


def exec_verify_sparql(query):
    status_code = 200
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    except SPARQLExceptions.EndPointInternalError:
        status_code = 500
    except SPARQLExceptions.QueryBadFormed:
        status_code = 400
    except SPARQLExceptions.EndPointNotFound:
        status_code = 404
    except SPARQLExceptions.Unauthorized:
        status_code = 401
    except SPARQLExceptions.URITooLong:
        status_code = 414
    except:
        status_code = -1

    pred_answer = []
    slot_values = []
    slot_idxes = []

    if status_code != 200:
        time.sleep(5)
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
        except:
            return pred_answer, slot_values, status_code
    status_code = 200

    # if len(results["results"]["bindings"]) > 0:
    #     slot_idxes = [k for k in results["results"]["bindings"][0] if "xsl" in k]
    slot_idxes = results["head"]["vars"]
    slot_idxes = [k for k in slot_idxes if "xsl" in k]
    print(slot_idxes)
    # print(query)
    for result in results["results"]["bindings"]:
        if 'x' in result:
            value = result["x"]["value"]
        elif 'callret-0' in result:
            value = result['callret-0']['value']
        elif 'value' in result:
            value = result['value']['value']
        else:
            raise Exception("UNKNOWN {}".format(result))
        if value.startswith(prefix):
            value = value[len(prefix):]
            value = value.replace("-08:00", '')
        values = [result[xsl]["value"] for xsl in slot_idxes]
        for i in range(len(values)):
            if values[i].startswith(prefix):
                values[i] = values[i][len(prefix):]
                values[i] = values[i].replace("-08:00", '')
        slot_values.append(values)
        pred_answer.append(value)
    # print(pred_answer)

    if len(slot_values) >= 10000:
        status_code = 100

    return pred_answer, slot_values, status_code


def exec_verify_sparql_xsl_only(query):
    status_code = 200
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
    except SPARQLExceptions.EndPointInternalError:
        status_code = 500
    except SPARQLExceptions.QueryBadFormed:
        status_code = 400
    except SPARQLExceptions.EndPointNotFound:
        status_code = 404
    except SPARQLExceptions.Unauthorized:
        status_code = 401
    except SPARQLExceptions.URITooLong:
        status_code = 414
    except:
        status_code = -1

    slot_values = []

    if status_code != 200:
        time.sleep(5)
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
        except:
            return slot_values, status_code
    status_code = 200

    slot_idxes = results["head"]["vars"]
    slot_idxes = [k for k in slot_idxes if "xsl" in k]
    # print(slot_idxes)
    # print(query)
    for result in results["results"]["bindings"]:
        values = [result[xsl]["value"] for xsl in slot_idxes]
        for i in range(len(values)):
            if values[i].startswith(prefix):
                values[i] = values[i][len(prefix):]
                values[i] = values[i].replace("-08:00", '')
        slot_values.append(values)
    # print(pred_answer)
    if len(slot_values) >= 10000:
        status_code = 100

    return slot_values, status_code


def exec_verify_sparql_xsl_only_fix_literal(query):
    if "^^<http://www.w3.org/2001/XMLSchema#" not in query:
        return exec_verify_sparql_xsl_only(query)
    slot_values, status_code = exec_verify_sparql_xsl_only(query)
    if status_code == 200 and len(slot_values) == 0:
        for data_type in ['integer', 'float', 'dateTime']:
            query = query.replace("^^<http://www.w3.org/2001/XMLSchema#{}>".format(data_type), "")
        slot_values, status_code = exec_verify_sparql_xsl_only(query)
    return slot_values, status_code


def exec_sparql_fix_literal(query):
    if "^^<http://www.w3.org/2001/XMLSchema#" not in query:
        return exec_sparql(query)
    answers, status_code = exec_sparql(query)
    if status_code == 200 and len(answers) == 0:
        query = re.sub("XMLSchema#\w+>", "", query.replace("^^<http://www.w3.org/2001/", ""))
        answers, status_code = exec_sparql(query)
    return answers, status_code


def retrieve_ent_meta_info(entities):
    query = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT DISTINCT ?ent_id ?ent_desc ?ent_type ?ent_name
            {
            VALUES ?ent_id {%s}
            OPTIONAL {?ent_id :type.object.name ?ent_name . FILTER (lang(?ent_name) = "en")}
            OPTIONAL {?ent_id :common.topic.description ?ent_desc . FILTER (lang(?ent_desc) = "en")} 
            OPTIONAL {?ent_id :kg.object_profile.prominent_type ?ent_type .}
            }
            """ % (" ".join([":{}".format(e) for e in entities]),)
    print(query)
    entity_meta_info, status_code = exec_demo_sparql(query)
    return entity_meta_info, status_code


def retrieve_schema_meta_info(schema_items):
    query = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT DISTINCT ?ent_id ?schema_id ?ent_desc ?ent_type ?ent_name
            {
            VALUES ?schema_id {%s}
            ?ent_id :type.object.key ?schema_id .
            OPTIONAL {?ent_id :type.object.name ?ent_name . FILTER (lang(?ent_name) = "en")} 
            OPTIONAL {?ent_id :common.topic.description ?ent_desc . FILTER (lang(?ent_desc) = "en")}
            OPTIONAL {?ent_id :kg.object_profile.prominent_type ?ent_type .}
            }
            """ % (" ".join(["\"/{}\"".format(e.replace(".", "/")) for e in schema_items]),)
    # print(query)
    entity_meta_info, status_code = exec_schema_demo_sparql(query)
    return entity_meta_info, status_code


def retrieve_anchor_relations(entities):
    query = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT DISTINCT ?ent_id ?in_rel
            {
            VALUES ?ent_id {%s}
            OPTIONAL {?s ?in_rel ?ent_id  . FILTER (regex(?in_rel, "^http://rdf.freebase.com/ns/"))}
            }
    """ % (" ".join([":{}".format(e) for e in entities]),)
    ent_in_relation, _ = exec_anchor_relation_sparql(query, "in_rel")
    query = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT DISTINCT ?ent_id ?out_rel
            {
            VALUES ?ent_id {%s}
            OPTIONAL {?ent_id ?out_rel ?v . FILTER (regex(?out_rel, "^http://rdf.freebase.com/ns/"))}
            }
    """ % (" ".join([":{}".format(e) for e in entities]),)
    ent_out_relation, _ = exec_anchor_relation_sparql(query, "out_rel")
    anchor_relations = {}
    for ent in entities:
        anchor_relations[ent] = {"in_relation": ent_in_relation.get(ent, []),
                                 "out_relation": ent_out_relation.get(ent, [])}
    return anchor_relations
