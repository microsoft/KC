# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

def string_to_bool(str):
    if isinstance(str, bool):
        return str
    if 'true' in str.lower():
        return True
    return False


def is_schema_in_lisp(schema: str, lisp: str):
    if (' ' + schema + ' ') in lisp:
        return True
    if (' ' + schema + ')') in lisp:
        return True
    return False


def get_entity_schema_in_lisp(lisp: str):
    entity = set()
    schema = set()
    if lisp is None or len(lisp) == 0:
        return entity, schema
    lisp_split = lisp.split(' ')
    for token in lisp_split:
        if token.startswith('('):
            continue
        if token.startswith('m.') or token.startswith('g.'):
            entity.add(token.strip(')'))
        else:
            schema.add(token.strip(')'))
    return entity, schema
