# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from overrides import overrides
from utils import extract_sxpression_structure, recursive_traverse_structure, structure_to_flatten_string
from .kb_world import KBWorld


class GrailKBWorld(KBWorld):

    """
    Private functions
    """
    @overrides
    def _preprocess(self, sexpression: str):
        """
        1. processing all functions
        2. distinguish JOIN_ENT and JOIN_REL
        """
        parsed_sexpression = re.sub(r'\bAND\b', 'AND_OP', sexpression)
        if 'JOIN' in sexpression:
            # build structure and identify whether a JOIN is JOIN_ENT or JOIN_REL
            parsed_sexpression = self._process_join_predicate(parsed_sexpression)
        return parsed_sexpression

    @overrides
    def postprocess(self, sexpression: str) -> str:
        """
        The reverse function for `preprocess`.
        """
        recover_sexpression = re.sub(r'\bAND_OP\b', 'AND', sexpression)
        recover_sexpression = re.sub(r'\bJOIN_ENT\b', 'JOIN', recover_sexpression)
        recover_sexpression = re.sub(r'\bJOIN_REL\b', 'JOIN', recover_sexpression)
        recover_sexpression = re.sub(r'\bJOIN_CLASS\b', 'JOIN', recover_sexpression)
        return recover_sexpression

    def _process_join_predicate(self, sexpression: str) -> str:
        expression_struc = extract_sxpression_structure(sexpression)
        results = []
        recursive_traverse_structure(expression_struc, keyword="JOIN", results=results)
        # reverse results since small clauses are visited later
        results = list(reversed(results))
        for i, clause in enumerate(results):
            assert len(clause) == 3
            # only check for the second argument
            if isinstance(clause[2], list):
                semantic_type = self.language.obtain_join_type(clause[2][0])
            else:
                semantic_type = self.language.obtain_join_type(clause[2])
            # modify the JOIN name
            results[i][0] = semantic_type
        # expression_struc has been modified now, and we should convert it back into string to parse
        return structure_to_flatten_string(expression_struc)
