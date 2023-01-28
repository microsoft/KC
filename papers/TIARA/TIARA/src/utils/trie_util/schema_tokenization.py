# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from transformers import T5Tokenizer

from retriever.freebase_retriever import FreebaseRetriever
from utils.config import class_token_path, relation_token_path, class_trie_path, relation_trie_path
from utils.domain_dict import fb_domain_dict
from utils.file_util import pickle_save, pickle_load
from utils.trie_util.trie import Trie


def schema_tokenization():
    """
    Tokenize the schema and save the tokenized schema to pickle files.
    Returns class token and relation token.
    -------

    """
    class_tokens = pickle_load(class_token_path)
    relation_tokens = pickle_load(relation_token_path)

    retriever = FreebaseRetriever()
    all_classes = set()
    all_relations = set()
    for domain in fb_domain_dict:
        for schema in fb_domain_dict[domain]:
            rdf_type = retriever.rdf_type_by_uri(schema)
            if rdf_type is not None:
                rdf_type = rdf_type[0]
            if 'Class' in rdf_type:
                all_classes.add(schema)
            elif 'Property' in rdf_type:
                all_relations.add(schema)
    with open('../dataset/grail_classes.txt', 'w') as f:
        for c in all_classes:
            print(c, file=f)
    with open('../dataset/grail_relations.txt', 'w') as f:
        for r in all_relations:
            print(r, file=f)

    if class_tokens is None:
        class_tokens = []
        for c in all_classes:
            tokens = tokenizer.tokenize(c)
            class_tokens.append({'schema': c, 'tokens': tokens})
        print('#class:', len(class_tokens))
        pickle_save(class_tokens, class_token_path)

    if relation_tokens is None:
        relation_tokens = []
        for r in all_relations:
            tokens = tokenizer.tokenize(r)
            relation_tokens.append({'schema': r, 'tokens': tokens})
        print('#relation:', len(relation_tokens))
        pickle_save(relation_tokens, relation_token_path)
    return class_tokens, relation_tokens


def entity_tokenization():
    entity_tokens = tokenizer.tokenize('m.0dm6r_h')
    print(entity_tokens)


def build_token_trie():
    class_trie = pickle_load(class_trie_path)
    relation_trie = pickle_load(relation_trie_path)

    class_tokens, relation_tokens = schema_tokenization()
    if class_trie is None:
        class_trie = Trie()
        for class_token in class_tokens:
            class_trie.insert(class_token['tokens'])
        pickle_save(class_trie, class_trie_path)

    if relation_trie is None:
        relation_trie = Trie()
        for relation_token in relation_tokens:
            relation_trie.insert(relation_token['tokens'])
        pickle_save(relation_trie, relation_trie_path)

    return class_trie, relation_trie


if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    build_token_trie()
    # entity_tokenization()
