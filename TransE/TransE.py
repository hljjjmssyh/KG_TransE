from data_process import data
import tensorflow as tf
from tqdm import tqdm
import numpy as np


class TransE:

    def __init__(self, kg: data, embedding_dim,
                 margin_value, batch_size, learning_rate, dis_fucnc):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        'define input from TransE'
        self.pos_triple = tf.placeholder(tf.int32, shape=[None, 3])
        self.neg_triple = tf.placeholder(tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(tf.float32, shape=[None])
        self.dis_func = dis_fucnc
        self.loss = None
        self.train_op = None
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        'define training params'
        self.entity_embedding = None
        self.relation_embedding = None
        self.training_graph()
        'define eval params'
        self.eval_triple = tf.placeholder(tf.int32, shape=[3])
        self.h_ind = None
        self.t_ind = None
        self.eval_graph()

    def training_graph(self):
        """ embeddings initialize """
        bound = 6 / (self.embedding_dim ** 0.5)  #*******
        with tf.variable_scope('embedding') as scope:
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[self.kg.num_entity, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[self.kg.num_relation, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
        with tf.name_scope('normalization') as scope:
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
        with tf.name_scope('embedding_lookup') as scope:
            pos_h = tf.nn.embedding_lookup(self.entity_embedding, self.pos_triple[:, 0])
            pos_t = tf.nn.embedding_lookup(self.entity_embedding, self.pos_triple[:, 1])
            pos_r = tf.nn.embedding_lookup(self.relation_embedding, self.pos_triple[:, 2])
            neg_h = tf.nn.embedding_lookup(self.entity_embedding, self.neg_triple[:, 0])
            neg_t = tf.nn.embedding_lookup(self.entity_embedding, self.neg_triple[:, 1])
            neg_r = tf.nn.embedding_lookup(self.relation_embedding, self.neg_triple[:, 2])
        with tf.name_scope('loss') as scope:
            pos_dis = pos_h + pos_r - pos_t
            neg_dis = neg_h + neg_r - neg_t
            if self.dis_func == 'L1':
                pos_score = tf.reduce_sum(tf.abs(pos_dis), axis=1)
                neg_score = tf.reduce_sum(tf.abs(neg_dis), axis=1)
            else:
                pos_score = tf.reduce_sum(tf.square(pos_dis), axis=1)
                neg_score = tf.reduce_sum(tf.square(neg_dis), axis=1)
            self.loss = tf.reduce_sum(tf.maximum(self.margin + pos_score - neg_score, 0))
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def training_run(self, session: tf.Session, epoch):
        print('-----Start training -----Epoch:' + str(epoch))
        epoch_loss = 0
        used_triple = 0
        for i in self.kg.next_batch(self.batch_size):
            pos_batch, neg_batch = self.kg.generator_pos_neg_triples(i)
            neg_batch = np.array(neg_batch)
            _, batch_loss = session.run(fetches=[self.train_op, self.loss],
                                        feed_dict={
                                            self.pos_triple: pos_batch,
                                            self.neg_triple: neg_batch,
                                            self.margin: [self.margin_value] *len(pos_batch)
                                        })
            epoch_loss += batch_loss
            used_triple += len(pos_batch)
            # print('used_triple:{}/{} triple_avg_loss:{}'.format(used_triple,
            #                                                     self.kg.num_training_triple,
            #                                                     epoch_loss / used_triple))
        print()
        print('epoch loss: {:.3f}'.format(epoch_loss))

    def eval_graph(self):
        with tf.name_scope('eval_look_up') as scope:
            h = tf.nn.embedding_lookup(self.entity_embedding, self.eval_triple[0])
            t = tf.nn.embedding_lookup(self.entity_embedding, self.eval_triple[1])
            r = tf.nn.embedding_lookup(self.relation_embedding, self.eval_triple[2])
        with tf.name_scope('predict_link') as scope:
            h_pred = self.entity_embedding + r - t
            t_pred = h + r - self.entity_embedding
            if self.dis_func == 'L1':
                dis_h = tf.reduce_sum(tf.abs(h_pred), axis=1)
                dis_t = tf.reduce_sum(tf.abs(t_pred), axis=1)
            else:
                dis_h = tf.reduce_sum(tf.square(h_pred), axis=1)
                dis_t = tf.reduce_sum(tf.square(t_pred), axis=1)
            _,self.h_idx = tf.nn.top_k(dis_h,k=self.kg.num_entity)
            _, self.t_idx = tf.nn.top_k(dis_t,k=self.kg.num_entity)

    def eval_run(self,sess:tf.Session):
        head_mean_bank_raw = 0
        head_hit_10 = 0
        tail_mean_bank_raw = 0
        tail_hit_10 =0
        head_mean_bank_filter =0
        tail_mean_bank_filter =0
        z = 0
        print('-----Start evaluation-----')
        for triple in self.kg.test_triples:
            #print(triple)
            h_idx, t_idx = sess.run(fetches=[self.h_idx,self.t_idx],
                                    feed_dict={
                                        self.eval_triple:triple
                                    })
            a,b,c,d,e,f = self.calculate_bank(h_idx,t_idx,triple)
            #print(triple)
            #print(a,b,c,d,e,f)
            head_mean_bank_raw += a
            head_mean_bank_filter +=b
            head_hit_10 +=c
            tail_mean_bank_raw +=d
            tail_mean_bank_filter +=e
            tail_hit_10 +=f
        print('-----Raw-----')
        print('-----head prediction-----')
        print('Meanbank:{:.3f},Hit@10:{:.3f}'.format(head_mean_bank_raw/len(self.kg.test_triples),
                                                     head_hit_10/len(self.kg.test_triples)))
        print('-----tail prediction-----')
        print('Meanbank:{:.3f},Hit@10:{:.3f}'.format(tail_mean_bank_raw/len(self.kg.test_triples),
                                                     tail_hit_10/len(self.kg.test_triples)))
    def calculate_bank(self, idx_head_prediction, idx_tail_prediction, eval_triple):
        head, tail, relation = eval_triple
        head_rank_raw = 0
        tail_rank_raw = 0
        head_rank_filter = 0
        tail_rank_filter = 0
        head_hit = 0
        tail_hit = 0
        idx_head_prediction =idx_head_prediction[::-1]
        idx_tail_prediction = idx_tail_prediction[::-1]
        for candidate in idx_head_prediction:
            if candidate == head:
                break
            else:
                head_rank_raw += 1
                if (candidate, tail, relation) in self.kg.triple_pool:
                    continue
                else:
                    head_rank_filter += 1
        for candidate in idx_tail_prediction:
            if candidate == tail:
                break
            else:
                tail_rank_raw += 1
                if (head, candidate, relation) in self.kg.triple_pool:
                    continue
                else:
                    tail_rank_filter += 1
        if head_rank_raw< 10:
            head_hit = 1
        if tail_rank_raw< 10:
            tail_hit = 1
        return head_rank_raw,head_rank_filter,head_hit,tail_rank_raw,tail_rank_filter,tail_hit

