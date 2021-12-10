# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List
from allennlp.common.checks import ConfigurationError
from allennlp.data import Tokenizer, Token
from allennlp.data.tokenizers import SpacyTokenizer
from utils import Class, Relation, NumberEntity, Universe, EntityType, EntityEncode

word_tokenizer = SpacyTokenizer()


class KnowledgeGraph:
    """
    A ``KnowledgeGraph`` represents a collection of entities and their relationships.

    The ``KnowledgeGraph`` currently stores (untyped) neighborhood information and text
    representations of each entity (if there is any).

    The knowledge base itself can be a table (like in WikitableQuestions), a figure (like in NLVR)
    or some other structured knowledge source. This abstract class needs to be inherited for
    implementing the functionality appropriate for a given KB.

    All of the parameters listed below are stored as public attributes.

    Parameters
    ----------
    entities : ``List[str]``
        The string identifiers of the entities in this knowledge graph.  We sort this set and store
        it as a list.  The sorting is so that we get a guaranteed consistent ordering across
        separate runs of the code.
    entity_text : ``Dict[str, str]``
        If you have additional text associated with each entity (other than its string identifier),
        you can store that here.  This might be, e.g., the text in a table cell, or the description
        of a wikipedia entity.
    """

    def __init__(self,
                 entities: List[str],
                 entity_text: Dict[str, str] = None,
                 entity_domain: List[str] = None) -> None:
        # from special entity information (including entity type, entity id, entity class) to index
        self.entities = entities
        # from the same key in entities to its entity text
        self.entity_text = entity_text
        # record each index corresponding domaion
        self.entity_domain = entity_domain

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


class KBContext:
    def __init__(self, tokenizer: Tokenizer,
                 utterance: List[Token],
                 # candidate entity & relation
                 entity_list: List[Class],
                 relation_list: List[Relation],
                 encode_method: str = "self",
                 encode_sep_word: str = "-",
                 truncate_entity_len: int = 15,
                 truncate_class_len: int = 20):

        self.tokenized_utterance = utterance
        self.truncate_entity_len = truncate_entity_len
        self.truncate_class_len = truncate_class_len

        # NOTICE: this knowledge graph is special since its entities include both entity & relation
        # the neighbors records the connection from an entity to an relation
        # these neighbors can be used in the input matching feature extraction
        # or the output semantic constrained decoding

        # manually build possible number candidates
        self.number_list = []
        for entity in entity_list:
            if entity.entity_type == EntityType.entity_num:
                self.number_list.append(
                    NumberEntity.entity_to_number(entity_id=entity.entity_id,
                                                  entity_class=entity.entity_class,
                                                  friendly_name=entity.text,
                                                  all_entity_classes=entity.all_entity_classes))
        self.entity_list = entity_list
        self.relation_list = relation_list
        # initialize index
        self.relation_start_index = 0
        # sep word is used to build hierarchy-aware representations
        self.encode_method = encode_method
        self.encode_sep_word = encode_sep_word.lower()

        # we could use key in knowledge_graph to search its related entity content
        self.knowledge_graph = self.get_knowledge_graph(entity_list, relation_list)

        entity_texts = [self.knowledge_graph.entity_text[entity].lower()
                        for entity in self.knowledge_graph.entities]
        entity_tokens = tokenizer.batch_tokenize(entity_texts)
        for i in range(len(entity_tokens)):
            entity_tokens[i] = [Token(text=t.text, lemma_=t.lemma_) if t.lemma_ != '-PRON-'
                                else Token(text=t.text, lemma_=t.text) for t in entity_tokens[i]]
        self.entity_tokens = entity_tokens

    @classmethod
    def get_entity_key(cls, entity: Class):

        coarse_entity_type = entity.entity_type.replace(":", "_")
        fine_grained_type = entity.entity_class.replace(":", "_")
        return f"{Universe.entity_repr}:{coarse_entity_type}:{fine_grained_type}:{entity.entity_id}"

    @classmethod
    def get_relation_key(cls, relation: Relation):
        # by default we use str type for single entity
        relation_type = relation.relation_class.lower().replace(":", "_")
        return f"{Universe.relation_repr}:{relation_type}:{relation.relation_id}"

    def get_entity_repr(self, ent: Class):
        replace_word = f" {self.encode_sep_word} " if self.encode_method in [EntityEncode.sep_top_domain,
                                                                             EntityEncode.sep_all_domain,
                                                                             EntityEncode.sep_close_domain] else " "
        # truncate the entity len
        try_tokens = word_tokenizer.tokenize(ent.text.lower())
        try_tokens = try_tokens[:self.truncate_entity_len]
        entity_text = " ".join([token.text for token in try_tokens])

        # no hierarchical relations
        if self.encode_method == EntityEncode.self or '.' not in ent.entity_class:
            # just return self
            encode_text = entity_text
        else:
            entity_classes = ent.entity_class.lower().replace("_", " ").split(".")
            # entity or entity class
            entity_self_text = entity_text if ent.entity_type in [EntityType.entity_str,
                                                                  EntityType.entity_num] else entity_classes[-1]
            if self.encode_method in [EntityEncode.all_domain, EntityEncode.sep_all_domain]:
                if ent.entity_type in [EntityType.entity_str, EntityType.entity_num]:
                    encode_text = replace_word.join(entity_classes + [entity_self_text])
                else:
                    encode_text = replace_word.join(entity_classes)
            elif self.encode_method in [EntityEncode.top_domain, EntityEncode.sep_top_domain]:
                # only retain the top domain and self text
                encode_text = replace_word.join([entity_classes[0], entity_self_text])
            elif self.encode_method in [EntityEncode.close_domain, EntityEncode.sep_close_domain]:
                # only retain the top domain and self text
                if ent.entity_type in [EntityType.entity_str, EntityType.entity_num]:
                    encode_text = replace_word.join([entity_classes[-1], entity_self_text])
                else:
                    encode_text = replace_word.join([entity_classes[-2], entity_self_text])
            else:
                raise ConfigurationError("we do not support entity encode method as :{}".format(self.encode_method))

        encode_text = self.cut_class_len(encode_text, length=self.truncate_class_len
                                                             + self.truncate_entity_len)

        return encode_text

    def get_entity_domain(self, ent: Class):
        entity_class = " ".join(ent.entity_class.lower().replace("_", " ").split(".")[-2:])
        return entity_class

    def cut_class_len(self, text, length):
        try_tokens = word_tokenizer.tokenize(text)
        # keep the last ones which are mostly important
        try_tokens = try_tokens[- length:]
        entity_text = " ".join([token.text for token in try_tokens])
        return entity_text

    def get_relation_repr(self, rel: Relation):
        replace_word = f" {self.encode_sep_word} " if self.encode_method in [EntityEncode.sep_top_domain,
                                                                             EntityEncode.sep_all_domain,
                                                                             EntityEncode.sep_close_domain] else " "

        relation_classes = rel.relation_class.lower().replace("_", " ").split(".")
        # no hierarchical relations
        if self.encode_method == EntityEncode.self or '.' not in rel.relation_class:
            # just return self
            encode_text = rel.text.lower()
        elif self.encode_method in [EntityEncode.all_domain, EntityEncode.sep_all_domain]:
            encode_text = replace_word.join(relation_classes)
        elif self.encode_method in [EntityEncode.top_domain, EntityEncode.sep_top_domain]:
            # only retain the top domain and self text
            encode_text = replace_word.join([relation_classes[0], relation_classes[-1]])
        elif self.encode_method in [EntityEncode.close_domain, EntityEncode.sep_close_domain]:
            # only retain the top domain and self text
            encode_text = replace_word.join([relation_classes[-2], relation_classes[-1]])
        else:
            raise ConfigurationError("we do not support entity encode method as :{}".format(self.encode_method))

        encode_text = self.cut_class_len(encode_text, length=self.truncate_class_len)

        return encode_text

    def get_relation_domain(self, rel: Relation):
        relation_class = " ".join(rel.relation_class.lower().replace("_", " ").split(".")[-2:])
        return relation_class

    def get_knowledge_graph(self, class_list: List[Class],
                            relation_list: List[Relation]) -> KnowledgeGraph:
        entities: List[str] = []
        entity_domain: List[str] = []
        entity_text: Dict[str, str] = {}

        for entity in class_list:
            entity_key = self.get_entity_key(entity)
            entities.append(entity_key)
            entity_domain.append(self.get_entity_domain(entity))

            entity_text[entity_key] = self.get_entity_repr(entity)

        self.relation_start_index = len(entities)
        for relation in relation_list:
            relation_key = self.get_relation_key(relation)
            entities.append(relation_key)
            entity_domain.append(self.get_relation_domain(relation))
            entity_text[relation_key] = self.get_relation_repr(relation)

        kg = KnowledgeGraph(entities, entity_text, entity_domain)

        return kg

