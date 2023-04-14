import os
import json
import random

import torch
import torch.utils.data.dataset

from typing import Optional, List

from config import args
from triplet import reverse_triplet
from triplet_mask import construct_mask, construct_self_negative_mask, construct_rule_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer
from logger_config import logger

entity_dict = get_entity_dict()

# relation_dict = dict()
# relation_data = json.load(open(os.path.join(os.path.dirname(args.train_path), 'relations.json'), 'r', encoding='UTF-8'))
# for i, (relation_origin, relation_desc) in enumerate(relation_data.items()):
#     relation_desc = relation_desc.replace('!', 'inverse')
#     relation_dict[relation_desc] = i
relation_dict = dict()
desc2relation = dict()
with open(os.path.join(os.path.dirname(args.train_path), 'relations.dict'), 'r', encoding='UTF-8') as f:
    for line in f:
        line = line.strip().split('\t')
        relation_dict[line[1]] = int(line[0])
relation_json = json.load(open(os.path.join(os.path.dirname(args.train_path), 'relations.json'), 'r', encoding='UTF-8'))
for i, (relation_origin, relation_desc) in enumerate(relation_json.items()):
    relation_desc = relation_desc.strip()
    desc2relation[relation_desc] = relation_origin


def get_relation_id_by_desc(relation_desc):
    relation_origin = desc2relation[relation_desc]
    relation_id = relation_dict[relation_origin]
    return relation_id


if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)


class Example:

    def __init__(self, head_id, relation, tail_id, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)
        hr_encoded_inputs = _custom_tokenize(text=head_text,
                                             text_pair=self.relation)
        # hr_encoded_inputs = _custom_tokenize(text=head_text)

        head_encoded_inputs = _custom_tokenize(text=head_text)

        tail_word = _parse_entity_name(self.tail)
        tail_encoded_inputs = _custom_tokenize(text=_concat_name_desc(tail_word, tail_desc))
        # tail_encoded_inputs = _custom_tokenize(text=tail_word)

        if 'inverse' in self.relation:
            self.relation = self.relation.replace('inverse', '')
        self.relation = self.relation.strip()

        if self.relation != '':
            relation_ids = [get_relation_id_by_desc(self.relation)]
        else:
            relation_ids = None

        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                'head_token_ids': head_encoded_inputs['input_ids'],
                'head_token_type_ids': head_encoded_inputs['token_type_ids'],
                'relation_ids': relation_ids,
                'obj': self}


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
         return self.examples[index].vectorize()


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False)

    batch_exs = [ex['obj'] for ex in batch_data]
    relation_ids = [ex['relation_ids'][0] for ex in batch_data if ex['relation_ids'] is not None]
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
        'relation_ids': relation_ids
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices


class RuleExample:

    def __init__(self, rule_head_id, rule_head_desc, rule_body_id, rule_body_desc):
        self.rule_head_id = int(rule_head_id)
        self.rule_head_desc = rule_head_desc
        self.rule_body_ids = [int(ele) for ele in rule_body_id.split()]
        self.rule_body_desc = rule_body_desc.replace('<sep>', ' ')

    def vectorize(self) -> dict:
        rule_head_encoded_inputs = _custom_tokenize(text=self.rule_head_desc)
        rule_body_encoded_inputs = _custom_tokenize(text=self.rule_body_desc)
        return {
            'rule_head_token_ids': rule_head_encoded_inputs['input_ids'],
            'rule_head_token_type_ids': rule_head_encoded_inputs['token_type_ids'],
            'rule_body_token_ids': rule_body_encoded_inputs['input_ids'],
            'rule_body_token_type_ids': rule_body_encoded_inputs['token_type_ids'],
            'obj': self
        }


class RuleDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, is_valid=False, examples=None):
        self.path_list = path.split(',')
        self.task = task
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                self.examples = load_rule_data(path, is_valid)
            else:
                self.examples.extend(load_rule_data(path, is_valid))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item].vectorize()


def load_rule_data(path: str, is_valid=False) -> List[RuleExample]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    logger.info('In test model: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='UTF-8'))
    data_len = len(data)
    # random.shuffle(data)
    rule_head_id_len = {}
    for ele in data:
        rule_head_id = ele.get('rule_head_id')
        if rule_head_id not in rule_head_id_len:
            rule_head_id_len[rule_head_id] = 0
        rule_head_id_len[rule_head_id] += 1

    train_data = {}
    valid_data = {}
    for ele in data:
        rule_head_id = ele.get('rule_head_id')
        if rule_head_id not in train_data:
            train_data[rule_head_id] = list()
        if rule_head_id not in valid_data:
            valid_data[rule_head_id] = list()
        if len(train_data[rule_head_id]) < int(rule_head_id_len[rule_head_id] * 0.8):
            train_data[rule_head_id].append(ele)
        else:
            valid_data[rule_head_id].append(ele)

    train_data = [ele for _, ele in train_data.items()]
    valid_data = [ele for _, ele in valid_data.items()]
    train_data_list = []
    valid_data_list = []
    for ele in train_data:
        train_data_list.extend(ele)
    for ele in valid_data:
        valid_data_list.extend(ele)

    random.shuffle(train_data_list)

    if is_valid:
        data = valid_data_list
    else:
        data = train_data_list

    examples = []
    for i, obj in enumerate(data):
        examples.append(RuleExample(**obj))

    return examples


def rule_collate(batch_data: List[dict]) -> dict:
    rule_head_token_ids, rule_head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['rule_head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    rule_head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['rule_head_token_type_ids']) for ex in batch_data],
        need_mask=False
    )

    rule_body_token_ids, rule_body_mask = to_indices_and_mask(
        [torch.LongTensor(ex['rule_body_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id
    )
    rule_body_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['rule_body_token_type_ids']) for ex in batch_data],
        need_mask=False
    )

    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'rule_head_token_ids': rule_head_token_ids,
        'rule_head_mask': rule_head_mask,
        'rule_head_token_type_ids': rule_head_token_type_ids,
        'rule_body_token_ids': rule_body_token_ids,
        'rule_body_mask': rule_body_mask,
        'rule_body_token_type_ids': rule_body_token_type_ids,
        'batch_data': batch_exs,
        'rule_mask': construct_rule_mask(row_exs=batch_exs) if not args.is_test else None,
        # 'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None
    }
    return batch_dict
