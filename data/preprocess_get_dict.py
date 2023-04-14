import os


file_name = 'FB15k-237'

relation_set = set()
entity_set = set()


with open(os.path.join(file_name, 'train.txt'), 'r', encoding='UTF-8') as f:
    for line in f:
        line = line.strip().split()
        entity_set.add(line[0])
        relation_set.add(line[1])
        entity_set.add(line[2])


relation_set = list(relation_set)
entity_set = list(entity_set)


with open(os.path.join(file_name, 'relations.dict'), 'w', encoding='UTF-8') as wf:
    for i, ele in enumerate(relation_set):
        wf.write(str(i) + '\t' + ele + '\n')


with open(os.path.join(file_name, 'entities.dict'), 'w', encoding='UTF-8') as wf:
    for i, ele in enumerate(entity_set):
        wf.write(str(i) + '\t' + ele + '\n')
