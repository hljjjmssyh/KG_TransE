import os
import numpy as np
import pandas as pd
import random


class data:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity_dict = {}  # file_format ：entity_name-id
        self.relation_dict = {}  # file_format : relation_name-id
        self.num_entity = 0
        self.num_relation = 0
        self.training_triples = []  # triple_format : (h, l ,t)
        self.num_training_triple = 0
        self.entity_id = []
        self.relation_id = []
        self.num_valid_triples = 0
        self.valid_triples = []
        self.test_triples = []
        self.num_test_triples = 0
        self.load_id_data()
        self.load_triple()
        self.training_pool = set(self.training_triples)
        self.triple_pool = set(self.training_triples + self.test_triples + self.valid_triples)

    def load_id_data(self):
        entity_filename = 'entity2id.txt'
        relation_filename = 'relation2id.txt'
        print('-----Loading entity-id data')
        entity_data = pd.read_table(os.path.join(self.data_dir, entity_filename), header=None)
        self.entity_dict = dict(zip(entity_data[0], entity_data[1]))  # convert raw_data to dict
        self.num_entity = len(self.entity_dict)
        self.entity_id = list(self.entity_dict.values())
        print('#entity: {}'.format(self.num_entity))
        print('-----Loading relation-id data')
        relation_data = pd.read_table(os.path.join(self.data_dir, relation_filename), header=None)
        self.relation_dict = dict(zip(relation_data[0], relation_data[1]))
        self.num_relation = len(self.relation_dict)
        self.relation_id = list(self.relation_dict.values())
        print('#relation: {}'.format(self.num_relation))

    def load_triple(self):
        training_filename = 'train.txt'
        valid_filename = 'valid.txt'
        test_filename = 'test.txt'
        print('-----Loading training triples-----')
        training_data = pd.read_table(os.path.join(self.data_dir, training_filename), header=None)
        self.training_triples = list(zip([self.entity_dict[h] for h in training_data[0]],
                                         [self.entity_dict[t] for t in training_data[1]],
                                         [self.relation_dict[r] for r in training_data[2]]))
        self.num_training_triple = len(self.training_triples)
        print('#training triple: {}'.format(self.num_training_triple))
        print('-----Loading validation triples-----')
        valid_data = pd.read_table(os.path.join(self.data_dir, valid_filename), header=None)
        self.valid_triples = list(zip([self.entity_dict[h] for h in valid_data[0]],
                                      [self.entity_dict[t] for t in valid_data[1]],
                                      [self.relation_dict[r] for r in valid_data[2]]))
        self.num_valid_triples = len(self.valid_triples)
        print('#validation triples: {}'.format(self.num_valid_triples))
        print('-----Loading test triples-----')
        test_data = pd.read_table(os.path.join(self.data_dir, test_filename), header=None)
        self.test_triples = list(zip([self.entity_dict[h] for h in test_data[0]],
                                     [self.entity_dict[t] for t in test_data[1]],
                                     [self.relation_dict[r] for r in test_data[2]]))
        self.num_test_triples = len(self.test_triples)
        print('#test triples: {}'.format(self.num_test_triples))

    def next_batch(self, batch_size):
        random_triples_list = np.random.permutation(self.training_triples)
        start = 0
        while start < self.num_training_triple:
            end = min(start + batch_size, self.num_training_triple)
            yield random_triples_list[start:end].copy()
            start = start + batch_size

    def generator_pos_neg_triples(self, pos_batch):
        neg_batch = []
        for x in pos_batch:
            h = x[0]
            t = x[1]
            prob = np.random.binomial(1, 0.5)
            while True:
                if prob <= 0.5:
                    h = random.choice(self.entity_id)
                else:
                    t = random.choice(self.entity_id)
                if (h, t, x[2]) not in self.training_pool:
                    break
            neg_batch.append([h, t, x[2]])
        return pos_batch, neg_batch
