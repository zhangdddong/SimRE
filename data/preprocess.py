import os
import json
import argparse
import multiprocessing as mp
from multiprocessing import Pool
from typing import List


parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--task', default='WN18RR', type=str, metavar='N',
                    help='dataset name: WN18RR or ...')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of workers')
parser.add_argument('--train-path', default='', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', default='', type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--test-path', default='', type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--rule-path', default='', type=str, metavar='N',
                    help='path to rule data')
args = parser.parse_args()


wn18rr_id2ent = {}
fb15k_id2ent = {}
fb15k_id2desc = {}
wiki5m_id2rel = {}
wiki5m_id2ent = {}
wiki5m_id2text = {}


def _truncate(text: str, max_len: int):
    return ' '.join(text.split()[:max_len])


def dump_all_rules(rule_path: str):

    id2relation = dict()
    with open(os.path.join(os.path.dirname(rule_path), 'relations.dict'), 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip().split()
            id2relation[line[0]] = line[1].replace('_', ' ').replace('!', ' ').strip()

    examples = []
    with open(rule_path, 'r', encoding='UTF-8') as f:
        for i, line in enumerate(f):
            line = line.strip().split()
            if len(line) < 1:
                continue
            elif len(line) == 1:
                rule_head = line[0]
                rule_body = line[0]
                rule_body_desc = []
                for ele in rule_body:
                    rule_body_desc.append(id2relation[ele])
            else:
                rule_head = line[0]
                rule_body = line[1:]
                rule_body_desc = []
                for ele in rule_body:
                    rule_body_desc.append(id2relation[ele])
            examples.append({
                'rule_head_id': rule_head,
                'rule_head_desc': id2relation[rule_head],
                'rule_body_id': ' '.join(rule_body),
                'rule_body_desc': '<sep>'.join(rule_body_desc)
            })

    json.dump(examples,
              open(os.path.join(os.path.dirname(rule_path), 'rules.json'), 'w', encoding='UTF-8'),
              ensure_ascii=False, indent=4)


def dump_all_entities(examples, out_path, id2text: dict):
    id2entity = {}
    relations = set()
    for ex in examples:
        head_id = ex['head_id']
        relations.add(ex['relation'])
        if head_id not in id2entity:
            id2entity[head_id] = {'entity_id': head_id,
                                  'entity': ex['head'],
                                  'entity_desc': id2text[head_id]}
        tail_id = ex['tail_id']
        if tail_id not in id2entity:
            id2entity[tail_id] = {'entity_id': tail_id,
                                  'entity': ex['tail'],
                                  'entity_desc': id2text[tail_id]}
    print('Get {} entities, {} relations in total'.format(len(id2entity), len(relations)))

    json.dump(list(id2entity.values()), open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


def _normalize_relations(examples: List[dict], normalize_fn, is_train: bool):
    def _check_sanity(relation_id_to_str: dict):
        # We directly use normalized relation string as a key for training and evaluation,
        # make sure no two relations are normalized to the same surface form
        relation_str_to_id = {}
        for rel_id, rel_str in relation_id_to_str.items():
            if rel_str is None:
                continue
            if rel_str not in relation_str_to_id:
                relation_str_to_id[rel_str] = rel_id
            elif relation_str_to_id[rel_str] != rel_id:
                assert False, 'ERROR: {} and {} are both normalized to {}' \
                    .format(relation_str_to_id[rel_str], rel_id, rel_str)
        return

    relation_id_to_str = {}
    for ex in examples:
        rel_str = normalize_fn(ex['relation'])
        relation_id_to_str[ex['relation']] = rel_str
        ex['relation'] = rel_str

    _check_sanity(relation_id_to_str)

    if is_train:
        out_path = '{}/relations.json'.format(os.path.dirname(args.train_path))
        with open(out_path, 'w', encoding='UTF-8') as w:
            json.dump(relation_id_to_str, w, ensure_ascii=False, indent=4)
            print('Save {} relations to {}'.format(len(relation_id_to_str), out_path))
        relation_dict_path = '{}/relations.dict'.format(os.path.dirname(args.train_path))
        with open(relation_dict_path, 'w', encoding='UTF-8') as w:
            for i, relation in enumerate(relation_id_to_str):
                w.write(str(i) + '\t' + relation + '\n')


def _process_line_wn18rr(line: str) -> dict:
    line = line.strip().split('\t')
    assert len(line) == 3, 'Expect 3 fields for {}'.format('\t'.join(line))
    head_id, relation, tail_id = line[0], line[1], line[2]
    _, head, _ = wn18rr_id2ent[head_id]
    _, tail, _ = wn18rr_id2ent[tail_id]
    example = {
        'head_id': head_id,
        'head': head,
        'relation': relation,
        'tail_id': tail_id,
        'tail': tail
    }
    return example


def _load_wn18rr_texts(text_path: str):
    global wn18rr_id2ent
    with open(text_path, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip().split('\t')
            assert len(line) == 3, 'Invalid line: {}'.format('\t'.join(line))
            entity_id, word, desc = line[0], line[1].replace('__', ''), line[2]
            wn18rr_id2ent[entity_id] = (entity_id, word, desc)
    print('Load {} entities from {}'.format(len(wn18rr_id2ent), text_path))


def preprocess_wn18rr(path):
    if not wn18rr_id2ent:
        _load_wn18rr_texts('{}/wordnet-mlj12-definitions.txt'.format(os.path.dirname(path)))
    with open(path, 'r', encoding='UTF-8') as f:
        pool = Pool(processes=args.workers)
        examples = pool.map(_process_line_wn18rr, f)
        pool.close()
        pool.join()

    _normalize_relations(examples, normalize_fn=lambda rel: rel.replace('_', ' ').strip(),
                         is_train=(path == args.train_path))

    if path == args.train_path:
        entity_dict_path = '{}/entities.dict'.format(os.path.dirname(args.train_path))
        with open(entity_dict_path, 'w', encoding='UTF-8') as w:
            for i, entity in enumerate(wn18rr_id2ent):
                w.write(str(i) + '\t' + entity + '\n')

    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='UTF-8'), ensure_ascii=False, indent=4)
    print('Save {} examples to {}'.format(len(examples), out_path))
    return examples


def _load_fb15k237_desc(path: str):
    global fb15k_id2desc
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        entity_id, desc = fs[0], fs[1]
        fb15k_id2desc[entity_id] = _truncate(desc, 50)
    print('Load {} entity descriptions from {}'.format(len(fb15k_id2desc), path))


def _load_fb15k237_wikidata(path: str):
    global fb15k_id2ent, fb15k_id2desc
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        entity_id, name = fs[0], fs[1]
        name = name.replace('_', ' ').strip()
        if entity_id not in fb15k_id2desc:
            print('No desc found for {}'.format(entity_id))
        fb15k_id2ent[entity_id] = (entity_id, name, fb15k_id2desc.get(entity_id, ''))
    print('Load {} entity names from {}'.format(len(fb15k_id2ent), path))


def _process_line_fb15k237(line: str) -> dict:
    fs = line.strip().split('\t')
    assert len(fs) == 3, 'Expect 3 fields for {}'.format(line)
    head_id, relation, tail_id = fs[0], fs[1], fs[2]

    _, head, _ = fb15k_id2ent[head_id]
    _, tail, _ = fb15k_id2ent[tail_id]
    example = {'head_id': head_id,
               'head': head,
               'relation': relation,
               'tail_id': tail_id,
               'tail': tail}
    return example


def _normalize_fb15k237_relation(relation: str) -> str:
    tokens = relation.replace('./', '/').replace('_', ' ').strip().split('/')
    dedup_tokens = []
    for token in tokens:
        if token not in dedup_tokens[-3:]:
            dedup_tokens.append(token)
    # leaf words are more important (maybe)
    relation_tokens = dedup_tokens[::-1]
    relation = ' '.join([t for idx, t in enumerate(relation_tokens)
                         if idx == 0 or relation_tokens[idx] != relation_tokens[idx - 1]])
    return relation


def preprocess_fb15k237(path):
    if not fb15k_id2desc:
        _load_fb15k237_desc('{}/FB15k_mid2description.txt'.format(os.path.dirname(path)))
    if not fb15k_id2ent:
        _load_fb15k237_wikidata('{}/FB15k_mid2name.txt'.format(os.path.dirname(path)))

    lines = open(path, 'r', encoding='utf-8').readlines()
    pool = Pool(processes=args.workers)
    examples = pool.map(_process_line_fb15k237, lines)
    pool.close()
    pool.join()

    _normalize_relations(examples, normalize_fn=_normalize_fb15k237_relation, is_train=(path == args.train_path))

    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('Save {} examples to {}'.format(len(examples), out_path))
    return examples

# wikidata5m
def _load_wiki5m_id2rel(path: str):
    global wiki5m_id2rel

    for line in open(path, 'r', encoding='utf-8'):
        fs = line.strip().split('\t')
        assert len(fs) >= 2, 'Invalid line: {}'.format(line.strip())
        rel_id, rel_text = fs[0], fs[1]
        rel_text = _truncate(rel_text, 10)
        wiki5m_id2rel[rel_id] = rel_text

    print('Load {} relations from {}'.format(len(wiki5m_id2rel), path))


def _load_wiki5m_id2ent(path: str):
    global wiki5m_id2ent
    for line in open(path, 'r', encoding='utf-8'):
        fs = line.strip().split('\t')
        assert len(fs) >= 2, 'Invalid line: {}'.format(line.strip())
        ent_id, ent_name = fs[0], fs[1]
        wiki5m_id2ent[ent_id] = _truncate(ent_name, 10)

    print('Load {} entity names from {}'.format(len(wiki5m_id2ent), path))


def _load_wiki5m_id2text(path: str, max_len: int = 30):
    global wiki5m_id2text
    for line in open(path, 'r', encoding='utf-8'):
        fs = line.strip().split('\t')
        assert len(fs) >= 2, 'Invalid line: {}'.format(line.strip())
        ent_id, ent_text = fs[0], ' '.join(fs[1:])
        wiki5m_id2text[ent_id] = _truncate(ent_text, max_len)

    print('Load {} entity texts from {}'.format(len(wiki5m_id2text), path))


def _has_none_value(ex: dict) -> bool:
    return any(v is None for v in ex.values())


def _process_line_wiki5m(line: str) -> dict:
    fs = line.strip().split('\t')
    assert len(fs) == 3, 'Invalid line: {}'.format(line.strip())
    head_id, relation_id, tail_id = fs[0], fs[1], fs[2]
    example = {'head_id': head_id,
               'head': wiki5m_id2ent.get(head_id, None),
               'relation': relation_id,
               'tail_id': tail_id,
               'tail': wiki5m_id2ent.get(tail_id, None)}
    return example


def preprocess_wiki5m(path: str, is_train: bool) -> List[dict]:
    if not wiki5m_id2rel:
        _load_wiki5m_id2rel(path='{}/wikidata5m_relation.txt'.format(os.path.dirname(path)))
    if not wiki5m_id2ent:
        _load_wiki5m_id2ent(path='{}/wikidata5m_entity.txt'.format(os.path.dirname(path)))
    if not wiki5m_id2text:
        _load_wiki5m_id2text(path='{}/wikidata5m_text.txt'.format(os.path.dirname(path)))

    lines = open(path, 'r', encoding='utf-8').readlines()
    pool = Pool(processes=args.workers)
    examples = pool.map(_process_line_wiki5m, lines)
    pool.close()
    pool.join()

    _normalize_relations(examples, normalize_fn=lambda rel_id: wiki5m_id2rel.get(rel_id, None), is_train=is_train)

    invalid_examples = [ex for ex in examples if _has_none_value(ex)]
    print('Find {} invalid examples in {}'.format(len(invalid_examples), path))
    if is_train:
        # P2439 P1962 P3484 do not exist in wikidata5m_relation.txt
        # so after filtering, there are 819 relations instead of 822 relations
        examples = [ex for ex in examples if not _has_none_value(ex)]
    else:
        # Even though it's invalid (contains null values), we should not change validation/test dataset
        print('Invalid examples: {}'.format(json.dumps(invalid_examples, ensure_ascii=False, indent=4)))

    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('Save {} examples to {}'.format(len(examples), out_path))
    return examples


def main():
    all_examples = []
    for path in [args.train_path, args.valid_path, args.test_path]:
        assert os.path.exists(path)
        print('Process {}...'.format(path))
        if args.task.lower() == 'wn18rr':
            all_examples += preprocess_wn18rr(path)
        elif args.task.lower() == 'fb15k237':
            all_examples += preprocess_fb15k237(path)
        elif args.task.lower() in ['wiki5m_trans', 'wiki5m_ind']:
            all_examples += preprocess_wiki5m(path, is_train=(path == args.train_path))
        else:
            assert False, 'Unknown task: {}'.format(args.task)

    if args.task.lower() == 'wn18rr':
        id2text = {k: v[2] for k, v in wn18rr_id2ent.items()}
    elif args.task.lower() == 'fb15k237':
        id2text = {k: v[2] for k, v in fb15k_id2ent.items()}
    elif args.task.lower() in ['wiki5m_trans', 'wiki5m_ind']:
        id2text = wiki5m_id2text
    else:
        assert False, 'Unknown task: {}'.format(args.task)

    dump_all_entities(all_examples,
                      out_path='{}/entities.json'.format(os.path.dirname(args.train_path)),
                      id2text=id2text)

    # Dealing rule data
    dump_all_rules(args.rule_path)

    print('Done')


if __name__ == '__main__':
    # --task wn18rr --train-path ./WN18RR/train.txt --valid-path ./WN18RR/valid.txt --test-path ./WN18RR/test.txt --rule-path ./WN18RR/mined_rules.txt
    # --task fb15k237 --train-path ./FB15k-237/train.txt --valid-path ./FB15k-237/valid.txt --test-path ./FB15k-237/test.txt --rule-path ./FB15k-237/mined_rules.txt
    # --task wiki5m_trans --train-path ./wiki5m_trans/train.txt --valid-path ./wiki5m_trans/valid.txt --test-path ./wiki5m_trans/test.txt --rule-path ./wiki5m_trans/mined_rules.txt
    # --task wiki5m_ind --train-path ./wiki5m_ind/train.txt --valid-path ./wiki5m_ind/valid.txt --test-path ./wiki5m_ind/test.txt --rule-path ./wiki5m_ind/mined_rules.txt
    main()
