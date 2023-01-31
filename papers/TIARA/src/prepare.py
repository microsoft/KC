# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Assumes working directory is: src

import os


def alter(file, old_str, new_str):
    file_data = ''
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str, new_str)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


def alter_lines(file, old_str, new_str):
    file_data = ''
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            file_data += line
    file_data = file_data.replace(old_str, new_str)
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


def comment(file, old_str):
    file_data = ''
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(' \n') == old_str:
                line = '# ' + line
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


if __name__ == '__main__':
    os.system('sh prepare.sh')

    os.system('wget https://raw.githubusercontent.com/dki-lab/GrailQA/main/ontology/domain_dict')
    os.system('mv domain_dict ../dataset/GrailQA/domain_dict.json')

    # download official GrailQA evaluation script (https://dki-lab.github.io/GrailQA/)
    os.system('wget -c "https://worksheets.codalab.org/rest/bundles/0x2d13989c17e44690ab62cc4edc0b900d/contents/blob/" -O grailqa_evaluate.py')
    alter('grailqa_evaluate.py', 'default=\'fb_roles\'', 'default=\'../dataset/GrailQA/ontology/fb_roles\'')
    alter('grailqa_evaluate.py', 'default=\'fb_types\'', 'default=\'../dataset/GrailQA/ontology/fb_types\'')
    alter('grailqa_evaluate.py', 'default=\'reverse_properties\'', 'default=\'../dataset/GrailQA/ontology/reverse_properties\'')
    os.system('mv grailqa_evaluate.py utils/statistics')

    os.system(
        'wget -c "https://raw.githubusercontent.com/dki-lab/GrailQA/27540d482db619212de0cebfa8859f67a9e9b7b1/entity_linking_results/value_extractor.py" -O utils/grailqa_value_extractor.py')

    os.system('wget https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/components/config.py')
    os.system('mv config.py retriever/components')
    os.system('wget https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/components/dataset_utils.py')
    os.system('mv dataset_utils.py retriever/components')
    os.system('wget https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/components/expr_parser.py')
    os.system('mv expr_parser.py retriever/components')
    os.system('wget https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/components/utils.py')
    os.system('mv utils.py retriever/components')

    os.system('wget https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/executor/cached_enumeration.py')
    os.system('mv cached_enumeration.py utils')
    alter('utils/cached_enumeration.py', '\'ontology/domain_info\'', 'os.path.dirname(__file__) + \'/../../dataset/GrailQA/ontology/domain_info\'')
    alter('utils/cached_enumeration.py', '\'ontology/fb_roles\'', 'os.path.dirname(__file__) + \'/../../dataset/GrailQA/ontology/fb_roles\'')
    alter('utils/cached_enumeration.py', '\'ontology/fb_types\'', 'os.path.dirname(__file__) + \'/../../dataset/GrailQA/ontology/fb_types\'')
    os.system('sed -i \'1i import os\' utils/cached_enumeration.py')

    os.system('wget https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/executor/logic_form_util.py')
    os.system('mv logic_form_util.py utils')
    alter('utils/logic_form_util.py', 'path + \'/../ontology/reverse_properties\'', '\'../dataset/GrailQA/ontology/reverse_properties\'')
    alter('utils/logic_form_util.py', 'path + \'/../ontology/fb_roles\'', '\'../dataset/GrailQA/ontology/fb_roles\'')
    alter('utils/logic_form_util.py', 'path + \'/../ontology/fb_types\'', '\'../dataset/GrailQA/ontology/fb_types\'')
    os.system('sed -i \'1i import sys\' utils/logic_form_util.py')
    alter('utils/logic_form_util.py', 'import sys', '\n'.join(['import sys', 'sys.path.append(\'.\')', 'sys.path.append(\'retriever\')']))
    alter('utils/logic_form_util.py', 'from executor.sparql_executor import execute_query', '')

    os.system('wget https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/GrailQA/run_disamb.py')
    os.system('mv run_disamb.py retriever')
    alter('retriever/run_disamb.py', 'do_predict=args.do_predict', 'do_predict=False')

    os.system('wget https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/GrailQA/scripts/run_disamb.sh')
    os.system('mv run_disamb.sh retriever/scripts')

    os.system('wget https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/framework/executor/sparql_executor.py')
    os.system('mv sparql_executor.py retriever/schema_linker')
    alter('retriever/schema_linker/sparql_executor.py', 'path + \'/../ontology/fb_roles\'', '\'../dataset/GrailQA/ontology/fb_roles\'')
    os.system('sed -i \'536i\ \ \ \ if entity is None or len(entity) == 0:\' retriever/schema_linker/sparql_executor.py')
    os.system('sed -i \'537i\ \ \ \ \ \ \ \ return in_relations, out_relations, paths\' retriever/schema_linker/sparql_executor.py')

    os.system('wget https://raw.githubusercontent.com/salesforce/rng-kbqa/2b6ef28e7724f11181f59589398894a1d0617455/GrailQA/detect_entity_mention.py')
    os.system('mv detect_entity_mention.py retriever')
    alter('retriever/detect_entity_mention.py', 'def get_all_entity_candidates(linker, utterance):', 'def get_all_entity_candidates(split, linker, utterance, qid):')
    alter('retriever/detect_entity_mention.py', 'all_candidates = get_all_entity_candidates(entity_linker, query)',
          'all_candidates, mentions = get_all_entity_candidates(args.split, entity_linker, query, qid)')
    alter('retriever/detect_entity_mention.py', '    mentions = linker.get_mentions(utterance)',
          '\n'.join(['    #SpanMD',
                     '    MD_file = f\'el_files/SpanMD_{split}.json\'',
                     '    with open(MD_file, \'r\', encoding=\'utf-8\') as f_mentions:',
                     '        all_mentions = json.load(f_mentions)',
                     '    mentions = all_mentions[qid]']))
    alter('retriever/detect_entity_mention.py', '    return all_entities', '    return all_entities, mentions')
    comment('retriever/detect_entity_mention.py', 'get_all_entity_candidates(entity_linker, "the music video stronger was directed by whom")')
    comment('retriever/detect_entity_mention.py', 'print')
    alter('retriever/detect_entity_mention.py', 'el_results[qid] = get_all_entity_candidates(entity_linker, query)',
          'el_results[qid] = get_all_entity_candidates(split, entity_linker, query)')
    comment('retriever/detect_entity_mention.py', 'sanity_checking = get_all_entity_candidates(entity_linker, "the music video stronger was directed by whom")')
    comment('retriever/detect_entity_mention.py', 'print(\'Checking result\', sanity_checking[0][:2])')
    alter('retriever/detect_entity_mention.py', 'datafile = f\'outputs/grailqa_v1.0_{split}.json\'',
          'datafile = f\'../../dataset/GrailQA/grailqa_v1.0_{split}.json\' if split != \'test\' else f\'../../dataset/GrailQA/grailqa_v1.0_test_public.json\'')

    comment('retriever/entity_linker/bert_entity_linker.py', 'from entity_linking.google_kg_api import get_entity_from_surface')
    alter('retriever/entity_linker/bert_entity_linker.py', 'from entity_linking', 'from entity_linker')
    alter('retriever/entity_linker/surface_index_memory.py', 'from entity_linking', 'from entity_linker')
    alter('retriever/entity_linker/aaqu_entity_linker.py', 'from entity_linking', 'from entity_linker')

    alter('retriever/models/RobertaRanker.py', 'from transformers.modeling_roberta import (', 'from transformers import (')

    alter('retriever/executor/sparql_executor.py', 'with open(path + \'/../ontology/fb_roles\', \'r\') as f:',
          'with open(os.path.dirname(__file__) + \'/../../../dataset/GrailQA/ontology/fb_roles\', \'r\') as f:')
    alter('retriever/executor/sparql_executor.py', 'with open(path + \'/../ontology/reverse_properties\', \'r\') as f:',
          'with open(os.path.dirname(__file__) + \'/../../../dataset/GrailQA/ontology/reverse_properties\', \'r\') as f:')
    os.system('sed -i \'1i import os\' retriever/executor/sparql_executor.py')

    alter('retriever/executor/logic_form_util.py', 'with open(path + \'/../ontology/fb_roles\', \'r\') as f:',
          'with open(os.path.dirname(__file__) + \'/../../../dataset/GrailQA/ontology/fb_roles\', \'r\') as f:')
    alter('retriever/executor/logic_form_util.py', 'with open(path + \'/../ontology/fb_types\', \'r\') as f:',
          'with open(os.path.dirname(__file__) + \'/../../../dataset/GrailQA/ontology/fb_types\', \'r\') as f:')
    alter('retriever/executor/logic_form_util.py', 'with open(path + \'/../ontology/reverse_properties\', \'r\') as f:',
          'with open(os.path.dirname(__file__) + \'/../../../dataset/GrailQA/ontology/reverse_properties\', \'r\') as f:')
    os.system('sed -i \'1i import os\' retriever/executor/logic_form_util.py')

    alter('retriever/models/BertRanker.py', 'return_dict = return_dict if return_dict is not None else self.config.use_return_dict', 'return_dict = False')

    alter('retriever/components/disamb_dataset.py', '    return relations_str',
          '\n'.join(['    return relations_str, len(relations)',
                     '',
                     'def _construct_candidate_type(args, tokenizer, cand_types):',
                     '    n_cand_types = [x for x in cand_types if x.split(\'.\')[0] not in _MODULE_DEFAULT.IGONORED_DOMAIN_LIST]',
                     '    cand_types_str = \' ; \'.join(map(_normalize_relation, n_cand_types))',
                     '    return cand_types_str'
                     ]))
    alter('retriever/components/disamb_dataset.py', '        relation_info = _construct_disamb_context(args, tokenizer, c, query_tokens)',
          '        relation_info, relation_num = _construct_disamb_context(args, tokenizer, c, query_tokens)')
    alter_lines('retriever/components/disamb_dataset.py',
                '\n'.join(['            # print(len(p.ca))',
                           '            if not do_predict:',
                           '                if (len(p.candidates) > 1) and p.target_id is not None:',
                           '                # if (len(p.candidates) > 0) and p.target_id is not None:',
                           '                    valid_disamb_problems.append(p)',
                           '                    if p.candidates[0].id == p.target_id:',
                           '                        baseline_acc += 1',
                           '            else:',
                           '                if (len(p.candidates) > 1):',
                           '                    valid_disamb_problems.append(p)'
                           ]),
                '            valid_disamb_problems.append(p)')
    alter('retriever/components/disamb_dataset.py', '            covered += 1',
          '\n'.join(['            covered += 1',
                     '    print(\'correct linked questions: {}\'.format(covered))',
                     '    print(\'total instances: \', len(instances))'
                     ]))

    alter('retriever/run_disamb.py', '    acc = np.sum(all_pred_indexes == all_labels) / len(all_pred_indexes)', '\n'.join([
        '    acc = np.sum(all_pred_indexes == all_labels) / len(all_pred_indexes)',
        '    np.set_printoptions(threshold=sys.maxsize)',
        '',
        '    print(\'correct linked: {}\'.format(np.sum(all_pred_indexes == all_labels)))',
        '    print(\'all_pred_indexes: \', len(all_pred_indexes))'
    ]))
    alter('retriever/run_disamb.py', '    if output_prediction:', '')
    alter('retriever/run_disamb.py',
          '        dump_json(OrderedDict([(feat.pid, pred) for feat, pred in zip(dataset, all_pred_indexes.tolist())]),',
          '    dump_json(OrderedDict([(feat.pid, pred) for feat, pred in zip(dataset, all_pred_indexes.tolist())]),',
          )

    alter('retriever/parse_sparql.py', '\'outputs/WebQSP.{split}.json\'', '\'../dataset/WebQSP/WebQSP.{split}.json\'')
    alter('retriever/parse_sparql.py', '\'outputs/WebQSP.{split}.expr.json\'', '\'../dataset/WebQSP/RnG/WebQSP.{split}.expr.json\'')
    alter('retriever/parse_sparql.py', '\'outputs/WebQSP.train.json\'', '\'../dataset/WebQSP/WebQSP.train.json\'')
    alter('retriever/parse_sparql.py', '\'outputs/WebQSP.ptrain.json\'', '\'../dataset/WebQSP/RnG/WebQSP.ptrain.json\'')
    alter('retriever/parse_sparql.py', '\'outputs/WebQSP.pdev.json\'', '\'../dataset/WebQSP/RnG/WebQSP.pdev.json\'')
    alter('retriever/parse_sparql.py', '\'outputs/WebQSP.train.expr.json\'', '\'../dataset/WebQSP/RnG/WebQSP.train.expr.json\'')
    alter('retriever/parse_sparql.py', '\'outputs/WebQSP.ptrain.expr.json\'', '\'../dataset/WebQSP/RnG/WebQSP.ptrain.expr.json\'')
    alter('retriever/parse_sparql.py', '\'outputs/WebQSP.pdev.expr.json\'', '\'../dataset/WebQSP/RnG/WebQSP.pdev.expr.json\'')
    os.system('python retriever/parse_sparql.py')

    alter('utils/cached_enumeration.py', 'class FBTwoHopPathCache(FBCacheBase):', '\n'.join(
        [
            'class FBTwoHopPathCache(FBCacheBase):',
            '    def query_two_hop_relations(self, entity):',
            '        two_hop_paths = self.query_two_hop_paths(entity)',
            '        res = set()',
            '        for path in two_hop_paths:',
            '            res.add(path[0].replace(\'#R\', \'\'))',
            '            res.add(path[1].replace(\'#R\', \'\'))',
            '        return res'
        ]))
    alter('retriever/enumerate_candidates.py', 'from executor.cached_enumeration', 'from utils.cached_enumeration')
