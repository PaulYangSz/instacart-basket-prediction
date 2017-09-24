import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_frame import DataFrame
from tf_base_model import TFBaseModel


class DataReader(object):  # 读取数据x和y，然后将这两列数据划分train和val

    def __init__(self, data_dir):
        data_cols = ['x', 'y']
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]

        df = DataFrame(columns=data_cols, data=data)

        self.train_df, self.val_df = df.train_test_split(train_size=0.9)

        print('train size', len(self.train_df))
        print('val size', len(self.val_df))

        self.num_products = df['x'].max() + 1  # 全部数据x中的商品ID最大值+1
        self.product_dist = np.bincount(self.train_df['x']).tolist()  # 按照x中的商品ID，从0到ID.max()进行统计个数，长度是训练集x中的商品ID最大值+1

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000  # 在数据集中往复的最大次数
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000  # 在数据集中往复的最大次数
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
        return df.batch_generator(batch_size, shuffle=shuffle, num_epochs=num_epochs)


class sgns(TFBaseModel):

    def __init__(self, embedding_dim=25, negative_samples=100, **kwargs):
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        super(sgns, self).__init__(**kwargs)

    def calculate_loss(self):
        self.x = tf.placeholder(dtype=tf.int32, shape=[None])  # x的输入，一次订单中商品y的附近商品ID
        self.y = tf.placeholder(dtype=tf.int32, shape=[None])  # y的输入，固定商品ID的值

        self.embeddings = tf.Variable(
            tf.random_uniform([self.reader.num_products, self.embedding_dim], -1.0, 1.0)  # [df['x'].max()+1, 25]
        )
        nce_weights = tf.Variable(
            tf.truncated_normal(
                shape=[self.reader.num_products, self.embedding_dim],  # [df['x'].max()+1, 25]
                stddev=1.0 / np.sqrt(self.embedding_dim)
            )
        )
        nce_biases = tf.Variable(tf.zeros([self.reader.num_products]))  # [df['x'].max()+1]

        inputs = tf.nn.embedding_lookup(self.embeddings, self.x)  # 当embedding_lookup()的第一个入参是单一的Tensor时和tf.gather等价
                                                                  # 当第一个入参是多个Tensor的list时，可以按照partition_strategy并行lookup，
        # 对于默认的'mod' strategy来说，假设第一个入参是[t1, t2, t3, ...]，那么相当于把这个Tensor列表按照如下方式展开成一个Tensor，然后再按照ids进行索引。
        # 展开成：t1[0], t2[0], t3[0], t1[1], t2[1], t3[1], ... ,这样结构的一个Tensor
        sampled_values = tf.nn.fixed_unigram_candidate_sampler(  # Samples a set of classes using the provided (fixed) base distribution.
            # 这里是指，对于y来说，每次只有一个值是正确的，然后在num_products的范围内采样negative_samples个负采样出来
            true_classes=tf.cast(tf.reshape(self.y, (-1, 1)), tf.int64),
            num_true=1,
            num_sampled=self.negative_samples,
            unique=True,
            range_max=self.reader.num_products,
            distortion=0.75,
            unigrams=self.reader.product_dist  # 给出了y的各个值的统计个数，也就是分布情况
        )

        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=self.y,
                inputs=inputs,
                num_sampled=self.negative_samples,
                num_classes=self.reader.num_products,
                sampled_values=sampled_values
            )
        )

        self.parameter_tensors = {
            'product_embeddings': self.embeddings
        }

        return loss


if __name__ == '__main__':
    base_dir = './'

    dr = DataReader(data_dir=os.path.join(base_dir, 'data'))

    sgns = sgns(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        learning_rate=.002,
        embedding_dim=25,  # ???
        negative_samples=100,  # ???
        batch_size=64,
        num_training_steps=10*10**6,
        early_stopping_steps=100000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        num_restarts=0,
        min_steps_to_checkpoint=100000,
        log_interval=500,
        num_validation_batches=4,  # 这个值*batch大小得到验证集的一次generator的数据量
        loss_averaging_window=5000,  # 记录train或者validation的loss历史窗长，在这里这个值比log_interval大，这样不会漏掉但会重复计算
        grad_clip=10,
    )
    sgns.fit()
    sgns.restore()
    sgns.predict()
