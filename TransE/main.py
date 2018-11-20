import tensorflow as tf
from TransE import TransE
from data_process import data
import argparse

def main():
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--data_dir',type=str,default='data/FB15k')
    parser.add_argument('--embedding_dim',type=int,default=100)
    parser.add_argument('--margin_value',type=float,default=1.0)
    parser.add_argument('--dis_func',type=str,default='L1')
    parser.add_argument('--batch_size',type=int,default=10000)
    parser.add_argument('--learning_rate',type=float,default=0.003)
    parser.add_argument('--max_epoch',type=int,default=2)
    args = parser.parse_args()
    print(args)
    kg = data(data_dir=args.data_dir)
    transE_model = TransE(kg=kg, embedding_dim=args.embedding_dim,margin_value=args.margin_value,
                          batch_size=args.batch_size,learning_rate=args.learning_rate,dis_fucnc=args.dis_func)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.max_epoch):
            transE_model.training_run(sess,epoch)
            if (epoch + 1) % 1 == 0:
                transE_model.eval_run(sess)
if __name__=='__main__':
    main()