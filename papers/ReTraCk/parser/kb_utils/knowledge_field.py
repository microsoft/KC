# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
``KnowledgeGraphField`` is a ``Field`` which stores a knowledge graph representation.
"""
from typing import List, Dict

from allennlp.data import TokenIndexer, Tokenizer
from allennlp_semparse.fields.knowledge_graph_field import KnowledgeGraphField
from allennlp.data.tokenizers.token import Token
from .kb_context import KnowledgeGraph
import editdistance
from overrides import overrides


class GrailKnowledgeGraphField(KnowledgeGraphField):
    """
    This implementation calculates all non-graph-related features (i.e. no related_column),
    then takes each one of the features to calculate related column features, by taking the max score of all neighbours
    """

    def __init__(self,
                 knowledge_graph: KnowledgeGraph,
                 utterance_tokens: List[Token],
                 token_indexers: Dict[str, TokenIndexer],
                 tokenizer: Tokenizer = None,
                 feature_extractors: List[str] = None,
                 entity_tokens: List[List[Token]] = None,
                 linking_features: List[List[List[float]]] = None,
                 include_in_vocab: bool = True,
                 max_table_tokens: int = None) -> None:
        feature_extractors = feature_extractors if feature_extractors is not None else [
            'exact_token_match',
            'contains_exact_token_match',
            # 'lemma_match',
            # 'contains_lemma_match',
            'edit_distance',
            'span_overlap_fraction'
            # 'span_lemma_overlap_fraction',
        ]

        super().__init__(knowledge_graph, utterance_tokens, token_indexers,
                         tokenizer=tokenizer, feature_extractors=feature_extractors, entity_tokens=entity_tokens,
                         linking_features=linking_features, include_in_vocab=include_in_vocab,
                         max_table_tokens=max_table_tokens)

    @overrides
    def _edit_distance(self,
                       entity: str,
                       entity_text: List[Token],
                       token: Token,
                       token_index: int,
                       tokens: List[Token]) -> float:
        entity_text = ' '.join(e.text for e in entity_text)
        edit_distance = float(editdistance.eval(entity_text, token.text))
        # normalize length
        maximum_len = max(len(entity_text), len(token.text))
        return 1.0 - edit_distance / maximum_len

    @overrides
    def empty_field(self) -> 'GrailKnowledgeGraphField':
        # TODO: HACK the error. We use utterance mask to judge whether the position is masked, not the KG field.
        return self
