# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List
import json


def lisp_to_nested_expression(lisp_string: str) -> List:
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]


def gen_xsl_expression(slot_idx):
    return " ".join(["?xsl{}".format(idx) for idx in range(slot_idx)])


def webqsp_lisp_to_verify_sparql(lisp_program: str):
    clauses = []
    order_clauses = []
    entities = set()  # collect entites for filtering
    # identical_variables = {}   # key should be smaller than value, we will use small variable to replace large variable
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    superlative = False
    superlative_arg2_slot = False
    superlative_sign = None
    superlative_arg2_slot_idx = -1

    if expression[0] in ['ARGMAX', 'ARGMIN']:
        superlative = True
        # remove all joins in relation chain of an arg function. In another word, we will not use arg function as
        # binary function here, instead, the arity depends on the number of relations in the second argument in the
        # original function
        if isinstance(expression[2], list):
            def retrieve_relations(exp: list):
                rtn = []
                for element in exp:
                    if element == 'JOIN':
                        continue
                    elif isinstance(element, str):
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'R':
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'JOIN':
                        rtn.extend(retrieve_relations(element))
                return rtn

            relations = retrieve_relations(expression[2])
            expression = expression[:2]
            expression.extend(relations)

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    count = False

    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    slot_idx = 0
    aux_idx = 0
    for i, subp in enumerate(sub_programs):
        i = str(i)
        if subp[0] == 'JOIN':
            if isinstance(subp[1], list):  # R relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + subp[2][1:] + " ns:" + subp[1][1] + " ?x" + i + " .")
                # ******SLOT******
                elif subp[2] == 'SLOT':
                    # existential variable // SLOT value
                    clauses.append("?xsl" + str(slot_idx) + " ns:" + subp[1][1] + " ?x" + i + " .")
                    slot_idx += 1
                # ******************
                elif " ".join(subp[2:])[0].isupper():
                    clauses.append("?sc" + str(aux_idx) + " ns:" + subp[1][1] + " ?x" + i + " .")
                    clauses.append("FILTER (str(?sc{}) = \"{}\")".format(aux_idx, " ".join(subp[2:])))
                    aux_idx += 1
                # elif subp[2] == "Country":
                #     clauses.append("?sc" + str(aux_idx)  + " ns:" + subp[1][1] + " ?x" + i  + " .")
                #     clauses.append("FILTER (str(?sc{}) = \"Country\")".format(aux_idx))
                #     aux_idx += 1
                # elif subp[2] == "Firearms":
                #     clauses.append("?sc" + str(aux_idx) + " ns:" + subp[1][1] + " ?x" + i + " .")
                #     clauses.append("FILTER (str(?sc{}) = \"Firearms\")".format(aux_idx))
                #     aux_idx += 1
                else:  # literal   (actually I think literal can only be object)
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime']:
                            subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0] + "-08:00", subp[2].split("^^")[1])
                        else:
                            subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0], subp[2].split("^^")[1])
                    clauses.append(subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
            else:
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + i + " ns:" + subp[1] + " ?x" + subp[2][1:] + " .")
                # ******SLOT******
                elif subp[2] == 'SLOT':
                    # existential variable // SLOT value
                    clauses.append("?x" + i + " ns:" + subp[1] + " ?xsl" + str(slot_idx) + " .")
                    slot_idx += 1
                # ******************
                elif " ".join(subp[2:])[0].isupper():
                    clauses.append("?x" + i + " ns:" + subp[1] + " " + "?sc" + str(aux_idx) + " .")
                    clauses.append("FILTER (str(?sc{}) = \"{}\")".format(aux_idx, " ".join(subp[2:])))
                    aux_idx += 1
                # elif subp[2] == "Country":
                #     clauses.append("?x" + i + " ns:" + subp[1] + " " + "?sc" + str(aux_idx) + " .")
                #     clauses.append("FILTER (str(?sc{}) = \"Country\")".format(aux_idx))
                #     aux_idx += 1
                # elif subp[2] == "Firearms":
                #     clauses.append("?x" + i + " ns:" + subp[1] + " " + "?sc" + str(aux_idx) + " .")
                #     clauses.append("FILTER (str(?sc{}) = \"Firearms\")".format(aux_idx))
                #     aux_idx += 1
                else:  # literal
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime']:
                            subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0] + "-08:00", subp[2].split("^^")[1])
                        else:
                            subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0], subp[2].split("^^")[1])
                    clauses.append("?x" + i + " ns:" + subp[1] + " " + subp[2] + " .")
        elif subp[0] == 'AND':
            var1 = int(subp[2][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            # identical_variables[var1] = int(i)
            if subp[1][0] == "#":
                var2 = int(subp[1][1:])
                root2 = get_root(var2)
                # identical_variables[var2] = int(i)
                if root1 > root2:
                    # identical_variables[var2] = var1
                    identical_variables_r[root1] = root2
                else:
                    # identical_variables[var1] = var2
                    identical_variables_r[root2] = root1
            else:  # 2nd argument is a class
                if subp[1] == "male":
                    clauses.append("?x" + i + " ns:people.person.gender ns:m.05zppz" + " .")
                elif subp[1] == "female":
                    clauses.append("?x" + i + " ns:people.person.gender ns:m.02zsn" + " .")
                else:
                    clauses.append("?x" + i + " ns:type.object.type ns:" + subp[1] + " .")
        elif subp[0] in ['le', 'lt', 'ge', 'gt']:  # the 2nd can only be numerical value
            clauses.append("?x" + i + " ns:" + subp[1] + " ?y" + i + " .")
            if subp[0] == 'le':
                op = "<="
            elif subp[0] == 'lt':
                op = "<"
            elif subp[0] == 'ge':
                op = ">="
            else:
                op = ">"
            if subp[2].__contains__('^^'):
                data_type = subp[2].split("^^")[1].split("#")[1]
                if data_type not in ['integer', 'float', 'dateTime']:
                    subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0] + "-08:00", subp[2].split("^^")[1])
                else:
                    subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0], subp[2].split("^^")[1])
            clauses.append(f"FILTER (?y{i} {op} {subp[2]})")
        elif subp[0] == 'TC':
            var = int(subp[1][1:])
            # identical_variables[var] = int(i)
            rooti = get_root(int(i))
            root_var = get_root(var)
            if rooti > root_var:
                identical_variables_r[rooti] = root_var
            else:
                identical_variables_r[root_var] = rooti

            year = subp[3]
            if year.endswith("^^http://www.w3.org/2001/XMLSchema#gYear"):
                year = year.split('^^')[0]
            if year == 'NOW':
                from_para = '"2015-08-10"^^xsd:dateTime'
                to_para = '"2015-08-10"^^xsd:dateTime'
            else:
                from_para = f'"{year}-12-31"^^xsd:dateTime'
                to_para = f'"{year}-01-01"^^xsd:dateTime'

            clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2]} ?sk0}} || ')
            clauses.append(f'EXISTS {{?x{i} ns:{subp[2]} ?sk1 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk1) <= {from_para}) }})')
            if subp[2][-4:] == "from":
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk3 . ')
            else:  # from_date -> to_date
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk3 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk3) >= {to_para}) }})')

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            superlative = True
            if subp[1][0] == '#':
                var = int(subp[1][1:])
                rooti = get_root(int(i))
                root_var = get_root(var)
                # identical_variables[var] = int(i)
                if rooti > root_var:
                    identical_variables_r[rooti] = root_var
                else:
                    identical_variables_r[root_var] = rooti
            else:  # arg1 is class
                clauses.append(f'?x{i} ns:type.object.type ns:{subp[1]} .')

            if len(subp) == 3:
                if subp[2] == "SLOT":
                    # existential variable // SLOT value
                    clauses.append(f'?x{i} ?xsl{slot_idx} ?sk0 .')
                    # assume only one superlative operation
                    superlative_arg2_slot_idx = slot_idx
                    slot_idx += 1
                    superlative_arg2_slot = True
                else:
                    clauses.append(f'?x{i} ns:{subp[2]} ?sk0 .')
            elif len(subp) > 3:
                for j, relation in enumerate(subp[2:-1]):
                    if j == 0:
                        var0 = f'x{i}'
                    else:
                        var0 = f'c{j - 1}'
                    var1 = f'c{j}'
                    if isinstance(relation, list) and relation[0] == 'R':
                        clauses.append(f'?{var1} ns:{relation[1]} ?{var0} .')
                    else:
                        clauses.append(f'?{var0} ns:{relation} ?{var1} .')
                if subp[-1] == "SLOT":
                    clauses.append(f'?c{j} ?xsl{slot_idx} ?sk0 .')
                    # assume only one superlative operation
                    superlative_arg2_slot_idx = slot_idx
                    slot_idx += 1
                    superlative_arg2_slot = True
                else:
                    clauses.append(f'?c{j} ns:{subp[-1]} ?sk0 .')

            if subp[0] == 'ARGMIN':
                superlative_sign = "MIN"
            elif subp[0] == 'ARGMAX':
                superlative_sign = "MAX"
            if subp[0] == 'ARGMIN' and not superlative_arg2_slot:
                order_clauses.append("ORDER BY ?sk0")
            elif subp[0] == 'ARGMAX' and not superlative_arg2_slot:
                order_clauses.append("ORDER BY DESC(?sk0)")
            if not superlative_arg2_slot:
                order_clauses.append("LIMIT 1")
            else:
                order_clauses.append(f"GROUP BY ?xsl{superlative_arg2_slot_idx}")


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')

    if superlative:
        arg_clauses = clauses[:]

    for entity in entities:
        clauses.append(f'FILTER (?x != ns:{entity})')
    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count:
        clauses.insert(0, f"SELECT COUNT (DISTINCT ?x) {gen_xsl_expression(slot_idx)}")
    elif superlative:
        if superlative_arg2_slot:
            if superlative_sign == "MAX":
                clauses.insert(0, "{" + f"SELECT ?xsl{superlative_arg2_slot_idx} MAX(?sk0) as ?sk0")
            else:
                clauses.insert(0, "{" + f"SELECT ?xsl{superlative_arg2_slot_idx} MIN(?sk0) as ?sk0")
        else:
            # add select slot variable
            clauses.insert(0, "{SELECT ?sk0")
        clauses = arg_clauses + clauses
        clauses.insert(0, "WHERE {")
        clauses.insert(0, f"SELECT DISTINCT {gen_xsl_expression(slot_idx)} ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT {gen_xsl_expression(slot_idx)} ?x")
    clauses.insert(0, "PREFIX ns: <http://rdf.freebase.com/ns/>")

    clauses.append('}')
    clauses.extend(order_clauses)
    if superlative:
        clauses.append('}')
        clauses.append('}')

    # for clause in clauses:
    #     print(clause)

    return '\n'.join(clauses)


def webqsp_lisp_to_verify_sparql_xsl_only(lisp_program: str):
    clauses = []
    order_clauses = []
    entities = set()  # collect entites for filtering
    # identical_variables = {}   # key should be smaller than value, we will use small variable to replace large variable
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    superlative = False
    superlative_arg2_slot = False
    superlative_sign = None
    superlative_arg2_slot_idx = -1

    if expression[0] in ['ARGMAX', 'ARGMIN']:
        superlative = True
        # remove all joins in relation chain of an arg function. In another word, we will not use arg function as
        # binary function here, instead, the arity depends on the number of relations in the second argument in the
        # original function
        if isinstance(expression[2], list):
            def retrieve_relations(exp: list):
                rtn = []
                for element in exp:
                    if element == 'JOIN':
                        continue
                    elif isinstance(element, str):
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'R':
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'JOIN':
                        rtn.extend(retrieve_relations(element))
                return rtn

            relations = retrieve_relations(expression[2])
            expression = expression[:2]
            expression.extend(relations)

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    count = False

    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    slot_idx = 0
    aux_idx = 0
    for i, subp in enumerate(sub_programs):
        i = str(i)
        if subp[0] == 'JOIN':
            if isinstance(subp[1], list):  # R relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + subp[2][1:] + " ns:" + subp[1][1] + " ?x" + i + " .")
                # ******SLOT******
                elif subp[2] == 'SLOT':
                    # existential variable // SLOT value
                    clauses.append("?xsl" + str(slot_idx) + " ns:" + subp[1][1] + " ?x" + i + " .")
                    slot_idx += 1
                # ******************
                elif " ".join(subp[2:])[0].isupper():
                    clauses.append("?sc" + str(aux_idx) + " ns:" + subp[1][1] + " ?x" + i + " .")
                    clauses.append("FILTER (str(?sc{}) = \"{}\")".format(aux_idx, " ".join(subp[2:])))
                    aux_idx += 1
                else:  # literal   (actually I think literal can only be object)
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime']:
                            subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0] + "-08:00", subp[2].split("^^")[1])
                        else:
                            subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0], subp[2].split("^^")[1])
                    clauses.append(subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
            else:
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + i + " ns:" + subp[1] + " ?x" + subp[2][1:] + " .")
                # ******SLOT******
                elif subp[2] == 'SLOT':
                    # existential variable // SLOT value
                    clauses.append("?x" + i + " ns:" + subp[1] + " ?xsl" + str(slot_idx) + " .")
                    slot_idx += 1
                # ******************
                elif " ".join(subp[2:])[0].isupper():
                    clauses.append("?x" + i + " ns:" + subp[1] + " " + "?sc" + str(aux_idx) + " .")
                    clauses.append("FILTER (str(?sc{}) = \"{}\")".format(aux_idx, " ".join(subp[2:])))
                    aux_idx += 1
                else:  # literal
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime']:
                            subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0] + "-08:00", subp[2].split("^^")[1])
                        else:
                            subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0], subp[2].split("^^")[1])
                    clauses.append("?x" + i + " ns:" + subp[1] + " " + subp[2] + " .")
        elif subp[0] == 'AND':
            var1 = int(subp[2][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            # identical_variables[var1] = int(i)
            if subp[1][0] == "#":
                var2 = int(subp[1][1:])
                root2 = get_root(var2)
                # identical_variables[var2] = int(i)
                if root1 > root2:
                    # identical_variables[var2] = var1
                    identical_variables_r[root1] = root2
                else:
                    # identical_variables[var1] = var2
                    identical_variables_r[root2] = root1
            else:  # 2nd argument is a class
                if subp[1] == "male":
                    clauses.append("?x" + i + " ns:people.person.gender ns:m.05zppz" + " .")
                elif subp[1] == "female":
                    clauses.append("?x" + i + " ns:people.person.gender ns:m.02zsn" + " .")
                else:
                    clauses.append("?x" + i + " ns:type.object.type ns:" + subp[1] + " .")
        elif subp[0] in ['le', 'lt', 'ge', 'gt']:  # the 2nd can only be numerical value
            clauses.append("?x" + i + " ns:" + subp[1] + " ?y" + i + " .")
            if subp[0] == 'le':
                op = "<="
            elif subp[0] == 'lt':
                op = "<"
            elif subp[0] == 'ge':
                op = ">="
            else:
                op = ">"
            if subp[2].__contains__('^^'):
                data_type = subp[2].split("^^")[1].split("#")[1]
                if data_type not in ['integer', 'float', 'dateTime']:
                    subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0] + "-08:00", subp[2].split("^^")[1])
                else:
                    subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0], subp[2].split("^^")[1])
            clauses.append(f"FILTER (?y{i} {op} {subp[2]})")
        elif subp[0] == 'TC':
            var = int(subp[1][1:])
            # identical_variables[var] = int(i)
            rooti = get_root(int(i))
            root_var = get_root(var)
            if rooti > root_var:
                identical_variables_r[rooti] = root_var
            else:
                identical_variables_r[root_var] = rooti

            year = subp[3]
            if year.endswith("^^http://www.w3.org/2001/XMLSchema#gYear"):
                year = year.split('^^')[0]
            if year == 'NOW':
                from_para = '"2015-08-10"^^xsd:dateTime'
                to_para = '"2015-08-10"^^xsd:dateTime'
            else:
                from_para = f'"{year}-12-31"^^xsd:dateTime'
                to_para = f'"{year}-01-01"^^xsd:dateTime'

            clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2]} ?sk0}} || ')
            clauses.append(f'EXISTS {{?x{i} ns:{subp[2]} ?sk1 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk1) <= {from_para}) }})')
            if subp[2][-4:] == "from":
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk3 . ')
            else:  # from_date -> to_date
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk3 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk3) >= {to_para}) }})')

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            superlative = True
            if subp[1][0] == '#':
                var = int(subp[1][1:])
                rooti = get_root(int(i))
                root_var = get_root(var)
                # identical_variables[var] = int(i)
                if rooti > root_var:
                    identical_variables_r[rooti] = root_var
                else:
                    identical_variables_r[root_var] = rooti
            else:  # arg1 is class
                clauses.append(f'?x{i} ns:type.object.type ns:{subp[1]} .')

            if len(subp) == 3:
                if subp[2] == "SLOT":
                    # existential variable // SLOT value
                    clauses.append(f'?x{i} ?xsl{slot_idx} ?sk0 .')
                    # assume only one superlative operation
                    superlative_arg2_slot_idx = slot_idx
                    slot_idx += 1
                    superlative_arg2_slot = True
                else:
                    clauses.append(f'?x{i} ns:{subp[2]} ?sk0 .')
            elif len(subp) > 3:
                for j, relation in enumerate(subp[2:-1]):
                    if j == 0:
                        var0 = f'x{i}'
                    else:
                        var0 = f'c{j - 1}'
                    var1 = f'c{j}'
                    if isinstance(relation, list) and relation[0] == 'R':
                        clauses.append(f'?{var1} ns:{relation[1]} ?{var0} .')
                    else:
                        clauses.append(f'?{var0} ns:{relation} ?{var1} .')
                if subp[-1] == "SLOT":
                    clauses.append(f'?c{j} ?xsl{slot_idx} ?sk0 .')
                    # assume only one superlative operation
                    superlative_arg2_slot_idx = slot_idx
                    slot_idx += 1
                    superlative_arg2_slot = True
                else:
                    clauses.append(f'?c{j} ns:{subp[-1]} ?sk0 .')

            if subp[0] == 'ARGMIN':
                superlative_sign = "MIN"
            elif subp[0] == 'ARGMAX':
                superlative_sign = "MAX"
            if subp[0] == 'ARGMIN' and not superlative_arg2_slot:
                order_clauses.append("ORDER BY ?sk0")
            elif subp[0] == 'ARGMAX' and not superlative_arg2_slot:
                order_clauses.append("ORDER BY DESC(?sk0)")
            if not superlative_arg2_slot:
                order_clauses.append("LIMIT 1")
            else:
                order_clauses.append(f"GROUP BY ?xsl{superlative_arg2_slot_idx}")

        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')

    if superlative:
        arg_clauses = clauses[:]

    for entity in entities:
        clauses.append(f'FILTER (?x != ns:{entity})')
    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")

    assert slot_idx > 0, f"No SLOT in {lisp_program}"

    if count:
        clauses.insert(0, f"SELECT {gen_xsl_expression(slot_idx)}")
    elif superlative:
        if superlative_arg2_slot:
            if superlative_sign == "MAX":
                clauses.insert(0, "{" + f"SELECT ?xsl{superlative_arg2_slot_idx} MAX(?sk0) as ?sk0")
            else:
                clauses.insert(0, "{" + f"SELECT ?xsl{superlative_arg2_slot_idx} MIN(?sk0) as ?sk0")
        else:
            # add select slot variable
            clauses.insert(0, "{SELECT ?sk0")
        clauses = arg_clauses + clauses
        clauses.insert(0, "WHERE {")
        clauses.insert(0, f"SELECT DISTINCT {gen_xsl_expression(slot_idx)}")
    else:
        clauses.insert(0, f"SELECT DISTINCT {gen_xsl_expression(slot_idx)}")
    clauses.insert(0, "PREFIX ns: <http://rdf.freebase.com/ns/>")

    clauses.append('}')
    clauses.extend(order_clauses)
    if superlative:
        clauses.append('}')
        clauses.append('}')

    # for clause in clauses:
    #     print(clause)

    return '\n'.join(clauses)


def webqsp_lisp_to_sparql(lisp_program: str):
    clauses = []
    order_clauses = []
    entities = set()  # collect entites for filtering
    # identical_variables = {}   # key should be smaller than value, we will use small variable to replace large variable
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    superlative = False
    if expression[0] in ['ARGMAX', 'ARGMIN']:
        superlative = True
        # remove all joins in relation chain of an arg function. In another word, we will not use arg function as
        # binary function here, instead, the arity depends on the number of relations in the second argument in the
        # original function
        if isinstance(expression[2], list):
            def retrieve_relations(exp: list):
                rtn = []
                for element in exp:
                    if element == 'JOIN':
                        continue
                    elif isinstance(element, str):
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'R':
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'JOIN':
                        rtn.extend(retrieve_relations(element))
                return rtn

            relations = retrieve_relations(expression[2])
            expression = expression[:2]
            expression.extend(relations)

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    count = False

    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    aux_idx = 0
    for i, subp in enumerate(sub_programs):
        i = str(i)
        if subp[0] == 'JOIN':
            if isinstance(subp[1], list):  # R relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + subp[2][1:] + " ns:" + subp[1][1] + " ?x" + i + " .")
                elif " ".join(subp[2:])[0].isupper():
                    clauses.append("?sc" + str(aux_idx) + " ns:" + subp[1][1] + " ?x" + i + " .")
                    clauses.append("FILTER (str(?sc{}) = \"{}\")".format(aux_idx, " ".join(subp[2:])))
                    aux_idx += 1
                else:  # literal   (actually I think literal can only be object)
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime']:
                            subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0] + "-08:00", subp[2].split("^^")[1])
                        else:
                            subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0], subp[2].split("^^")[1])
                    clauses.append(subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
            else:
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + i + " ns:" + subp[1] + " ?x" + subp[2][1:] + " .")
                elif " ".join(subp[2:])[0].isupper():
                    clauses.append("?x" + i + " ns:" + subp[1] + " " + "?sc" + str(aux_idx) + " .")
                    clauses.append("FILTER (str(?sc{}) = \"{}\")".format(aux_idx, " ".join(subp[2:])))
                    aux_idx += 1
                else:  # literal
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime']:
                            subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0] + "-08:00", subp[2].split("^^")[1])
                        else:
                            subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0], subp[2].split("^^")[1])
                    clauses.append("?x" + i + " ns:" + subp[1] + " " + subp[2] + " .")
        elif subp[0] == 'AND':
            var1 = int(subp[2][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            # identical_variables[var1] = int(i)
            if subp[1][0] == "#":
                var2 = int(subp[1][1:])
                root2 = get_root(var2)
                # identical_variables[var2] = int(i)
                if root1 > root2:
                    # identical_variables[var2] = var1
                    identical_variables_r[root1] = root2
                else:
                    # identical_variables[var1] = var2
                    identical_variables_r[root2] = root1
            else:  # 2nd argument is a class
                if subp[1] == "male":
                    clauses.append("?x" + i + " ns:people.person.gender ns:m.05zppz" + " .")
                elif subp[1] == "female":
                    clauses.append("?x" + i + " ns:people.person.gender ns:m.02zsn" + " .")
                else:
                    clauses.append("?x" + i + " ns:type.object.type ns:" + subp[1] + " .")
        elif subp[0] in ['le', 'lt', 'ge', 'gt']:  # the 2nd can only be numerical value
            clauses.append("?x" + i + " ns:" + subp[1] + " ?y" + i + " .")
            if subp[0] == 'le':
                op = "<="
            elif subp[0] == 'lt':
                op = "<"
            elif subp[0] == 'ge':
                op = ">="
            else:
                op = ">"
            if subp[2].__contains__('^^'):
                data_type = subp[2].split("^^")[1].split("#")[1]
                if data_type not in ['integer', 'float', 'dateTime']:
                    subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0] + "-08:00", subp[2].split("^^")[1])
                else:
                    subp[2] = '"{}"^^<{}>'.format(subp[2].split("^^")[0], subp[2].split("^^")[1])
            clauses.append(f"FILTER (?y{i} {op} {subp[2]})")
        elif subp[0] == 'TC':
            var = int(subp[1][1:])
            # identical_variables[var] = int(i)
            rooti = get_root(int(i))
            root_var = get_root(var)
            if rooti > root_var:
                identical_variables_r[rooti] = root_var
            else:
                identical_variables_r[root_var] = rooti

            year = subp[3]
            if year.endswith("^^http://www.w3.org/2001/XMLSchema#gYear"):
                year = year.split('^^')[0]
            if year == 'NOW':
                from_para = '"2015-08-10"^^xsd:dateTime'
                to_para = '"2015-08-10"^^xsd:dateTime'
            else:
                from_para = f'"{year}-12-31"^^xsd:dateTime'
                to_para = f'"{year}-01-01"^^xsd:dateTime'

            clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2]} ?sk0}} || ')
            clauses.append(f'EXISTS {{?x{i} ns:{subp[2]} ?sk1 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk1) <= {from_para}) }})')
            if subp[2][-4:] == "from":
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk3 . ')
            else:  # from_date -> to_date
                clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk2}} || ')
                clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk3 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk3) >= {to_para}) }})')

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            superlative = True
            if subp[1][0] == '#':
                var = int(subp[1][1:])
                rooti = get_root(int(i))
                root_var = get_root(var)
                # identical_variables[var] = int(i)
                if rooti > root_var:
                    identical_variables_r[rooti] = root_var
                else:
                    identical_variables_r[root_var] = rooti
            else:  # arg1 is class
                clauses.append(f'?x{i} ns:type.object.type ns:{subp[1]} .')

            if len(subp) == 3:
                clauses.append(f'?x{i} ns:{subp[2]} ?sk0 .')
            elif len(subp) > 3:
                for j, relation in enumerate(subp[2:-1]):
                    if j == 0:
                        var0 = f'x{i}'
                    else:
                        var0 = f'c{j - 1}'
                    var1 = f'c{j}'
                    if isinstance(relation, list) and relation[0] == 'R':
                        clauses.append(f'?{var1} ns:{relation[1]} ?{var0} .')
                    else:
                        clauses.append(f'?{var0} ns:{relation} ?{var1} .')

                clauses.append(f'?c{j} ns:{subp[-1]} ?sk0 .')

            if subp[0] == 'ARGMIN':
                order_clauses.append("ORDER BY ?sk0")
            elif subp[0] == 'ARGMAX':
                order_clauses.append("ORDER BY DESC(?sk0)")
            order_clauses.append("LIMIT 1")


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')

    if superlative:
        arg_clauses = clauses[:]

    for entity in entities:
        clauses.append(f'FILTER (?x != ns:{entity})')
    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count:
        clauses.insert(0, f"SELECT COUNT DISTINCT ?x")
    elif superlative:
        clauses.insert(0, "{SELECT ?sk0")
        clauses = arg_clauses + clauses
        clauses.insert(0, "WHERE {")
        clauses.insert(0, f"SELECT DISTINCT ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT ?x")
    clauses.insert(0, "PREFIX ns: <http://rdf.freebase.com/ns/>")

    clauses.append('}')
    clauses.extend(order_clauses)
    if superlative:
        clauses.append('}')
        clauses.append('}')

    # for clause in clauses:
    #     print(clause)

    return '\n'.join(clauses)


def _linearize_lisp_expression(expression: list, sub_formula_id):
    sub_formulas = []
    for i, e in enumerate(expression):
        if isinstance(e, list) and e[0] != 'R':
            sub_formulas.extend(_linearize_lisp_expression(e, sub_formula_id))
            expression[i] = '#' + str(sub_formula_id[0] - 1)
    sub_formulas.append(expression)
    sub_formula_id[0] += 1
    return sub_formulas

if __name__ == "__main__":
    print(webqsp_lisp_to_sparql("(JOIN (R government.government_position_held.office_holder) (TC (AND (JOIN (R government.governmental_jurisdiction.governing_officials) m.05kkh) (JOIN government.government_position_held.basic_title m.0fkvn)) government.government_position_held.from 2011^^http://www.w3.org/2001/XMLSchema#gYear))"))