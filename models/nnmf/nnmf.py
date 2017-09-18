import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_frame import DataFrame
from tf_base_model import TFBaseModel


class DataReader(object):
    """
    self.train_df  训练集，自定义的DataFrame类型
    self.val_df  验证集，自定义的DataFrame类型
    self.num_users
    self.num_products
    """

    def __init__(self, data_dir):
        # 读取prior中用户的id(i)，商品的id(j)以及对应的count(V_ij)值
        data_cols = ['i', 'j', 'V_ij']
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]

        df = DataFrame(columns=data_cols, data=data)  # 这里的DataFrame不是pandas的，是作者自己定义的
        self.train_df, self.val_df = df.train_test_split(train_size=0.9)

        print('train size', len(self.train_df))
        print('val size', len(self.val_df))

        self.num_users = df['i'].max() + 1  # TODO 有必要+1吗
        self.num_products = df['j'].max() + 1  # 有必要+1吗

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
        return df.batch_generator(batch_size, shuffle=shuffle, num_epochs=num_epochs, allow_smaller_final_batch=is_test)


class nnmf(TFBaseModel):

    def __init__(self, rank=25, **kwargs):
        self.rank = rank
        super(nnmf, self).__init__(**kwargs)

    def calculate_loss(self):
        """
        有两个矩阵，一个是W（和用户相关）一个是H（和商品相关），以及对应的列向量W_bias和H_bias
        Returns:

        """
        self.i = tf.placeholder(dtype=tf.int32, shape=[None])  # 输入，对于users的选择
        self.j = tf.placeholder(dtype=tf.int32, shape=[None])  # 输入，对于products的选择
        self.V_ij = tf.placeholder(dtype=tf.float32, shape=[None])  # 输入，对应count的值

        self.W = tf.Variable(tf.truncated_normal([self.reader.num_users, self.rank]))  # 学习的参数W，大小为num_users * rank的截断正态分布取值的tensor
        self.H = tf.Variable(tf.truncated_normal([self.reader.num_products, self.rank]))  # 学习的参数H，num_products * rank的截断正态分布取值的tensor
        W_bias = tf.Variable(tf.truncated_normal([self.reader.num_users]))  # 学习的参数，大小为num_users的截断正态分布取值的tensor
        H_bias = tf.Variable(tf.truncated_normal([self.reader.num_products]))  # 学习的参数，大小为num_products的截断正态分布取值的tensor

        global_mean = tf.Variable(0.0)   #
        w_i = tf.gather(self.W, self.i)  # 获得W的对应选择user的子集
        h_j = tf.gather(self.H, self.j)  # 获得H的对应选择product的子集

        w_bias = tf.gather(W_bias, self.i)  # 获得w_bias的对应选择user的子集
        h_bias = tf.gather(H_bias, self.j)  # 获得h_bias的对应选择product的子集
        interaction = tf.reduce_sum(w_i * h_j, reduction_indices=1)  # w和h的对应元素的乘积之后按行求和
        preds = global_mean + w_bias + h_bias + interaction  # 这些列向量相加

        rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(preds, self.V_ij)))  # preds和实际count的差异平方的均值开方

        self.parameter_tensors = {
            'user_embeddings': self.W,
            'product_embeddings': self.H
        }

        return rmse


if __name__ == '__main__':
    base_dir = './'

    # 读取models\nnmf\prepare_nnmf_data.py产生的数据文件，得到含有train和val以及可以生成batch子集的类DataReader
    dr = DataReader(data_dir=os.path.join(base_dir, 'data'))

    nnmf = nnmf(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        learning_rate=.005,
        rank=25,
        batch_size=4096,
        num_training_steps=150000,
        early_stopping_steps=30000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        num_restarts=0,
        min_steps_to_checkpoint=5000,
        log_interval=200,
        num_validation_batches=1,
        loss_averaging_window=200,

    )
    nnmf.fit()
    nnmf.restore()
    nnmf.predict()
