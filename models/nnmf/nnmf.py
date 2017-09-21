import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_frame import DataFrame
from tf_base_model import TFBaseModel


class DataReader(object):
    """
    self.train_df  训练集，自定义的DataFrame类型，由['i', 'j', 'V_ij']三个matrix构成，占所有data的0.9
    self.val_df  验证集，自定义的DataFrame类型，由['i', 'j', 'V_ij']三个matrix构成，占所有data的0.1
    self.num_users  用户ID的最大值+1
    self.num_products  商品ID的最大值+1
    """

    def __init__(self, data_dir):
        # 读取prior中用户的id(i)，商品的id(j)以及对应的count(V_ij)值
        data_cols = ['i', 'j', 'V_ij']
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i)), mmap_mode='r') for i in data_cols]

        df = DataFrame(columns=data_cols, data=data)  # 这里的DataFrame不是pandas的，是作者自己定义的，主要由colmumns（字符串list）和data（np.ndarray的list）组成
        self.train_df, self.val_df = df.train_test_split(train_size=0.9)

        print('train size', len(self.train_df))
        print('val size', len(self.val_df))

        self.num_users = df['i'].max() + 1  # TODO 有必要+1吗, W的shape=[num_users, nnmf.rank]，完全有必要，因为train和val中的数值都是1开始，而W的索引是从0开始，方便后面做tf.gather()
        self.num_products = df['j'].max() + 1  # 有必要+1吗, H的shape=[num_users, nnmf.rank]

    def train_batch_generator(self, batch_size):  # 取出batch大小的train数据集的子集
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
    """
    目的就是把用户IDs和商品IDs组成的以count为值的矩阵，做NNMF分解？最后得到两个矩阵W和H，其中W大小是[用户ID数+1，rank]，H大小是[商品ID数+1，rank]
    """

    def __init__(self, rank=25, **kwargs):
        self.rank = rank
        super(nnmf, self).__init__(**kwargs)

    def calculate_loss(self):
        """
        有两个矩阵，一个是W（和用户相关）一个是H（和商品相关），以及对应的列向量W_bias和H_bias
        预测值 = global_mean + w的bias + h的bias + (w * h)之后的按行求和，其中小写的变量都是对大写的变量代表的矩阵或者向量的取局部
        Returns: 预测的值和实际count的值的均方根误差

        """
        self.i = tf.placeholder(dtype=tf.int32, shape=[None])  # 输入，命名i对应自定义DataFrame的dict中的key方便在fit()中定义feed_dict，对于users的选择
        self.j = tf.placeholder(dtype=tf.int32, shape=[None])  # 输入，对于products的选择，shape=[batch, 1]，其中ID可重复
        self.V_ij = tf.placeholder(dtype=tf.float32, shape=[None])  # 输入，对应count的值，shape=[batch, 1]

        self.W = tf.Variable(tf.truncated_normal([self.reader.num_users, self.rank]))  # 学习的参数W，大小为num_users * rank的截断正态分布取值的tensor
        self.H = tf.Variable(tf.truncated_normal([self.reader.num_products, self.rank]))  # 学习的参数H，num_products * rank的截断正态分布取值的tensor
        W_bias = tf.Variable(tf.truncated_normal([self.reader.num_users]))  # 学习的参数，大小为num_users的截断正态分布取值的tensor
        H_bias = tf.Variable(tf.truncated_normal([self.reader.num_products]))  # 学习的参数，大小为num_products的截断正态分布取值的tensor

        global_mean = tf.Variable(0.0)   #
        w_i = tf.gather(self.W, self.i)  # i的取值为可重复的user_id的index，获得W的对应可重复行的batch*rank大小的矩阵
        h_j = tf.gather(self.H, self.j)  # j的取值为可重复的product_id的index，获得H的对应可重复行的batch*rank大小的矩阵

        w_bias = tf.gather(W_bias, self.i)  # 获得w_bias的对应选择user的子集
        h_bias = tf.gather(H_bias, self.j)  # 获得h_bias的对应选择product的子集
        interaction = tf.reduce_sum(w_i * h_j, reduction_indices=1)  # w和h的对应元素的直接相乘之后按行求和
        preds = global_mean + w_bias + h_bias + interaction  # 这些列向量相加

        rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(preds, self.V_ij)))  # preds和实际count的差异平方的均值开方

        self.parameter_tensors = {  # 这是后面在predict()中要将值保存到文件的tensor，其余的则在学习之后没有保存
            'user_embeddings': self.W,
            'product_embeddings': self.H
        }

        return rmse


if __name__ == '__main__':
    base_dir = './'

    # 读取models\nnmf\prepare_nnmf_data.py产生的数据文件，得到含有train和val以及可以生成batch子集的类DataReader
    # 其中train数据占比0.9, val数据占比0.1
    dr = DataReader(data_dir=os.path.join(base_dir, 'data'))

    nnmf = nnmf(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        learning_rate=.005,
        rank=25,  # 矩阵W和H的列数
        batch_size=4096,  # 对于训练集和验证集，取一次batch的大小
        num_training_steps=150000,  # 总共最大的训练次数
        early_stopping_steps=30000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,  # 是否使用tf.train.ExponentialMovingAverage
        num_restarts=0,  # 当early_stopping_steps满足时，可以降低学习率继续重试的最大次数
        min_steps_to_checkpoint=5000,  # 最少保存checkpoint的步数，而且必须是log_interval的整数倍，当它>num_training_steps时，在没有early_stopping时会最后保存，否则一定是
        log_interval=200,  # 每隔200步就记录下当前的val上的loss平均值，来看是否要early_stopping或者更新最佳loss及步数。
        num_validation_batches=1,  # 这个值*batch大小得到验证集上一次generator的数据量
        loss_averaging_window=200,  # 记录train或者validation的loss历史窗长，在这里这个值和log_interval相等，这样不会漏掉也不会重复计算

    )
    nnmf.fit()
    nnmf.restore()
    nnmf.predict()  # 最终得到W[user_id.max + 1, rank=25]和H[product_id.max + 1, rank=25]两个矩阵的数据文件，其中W和H的元素相乘再按行相加后再加上bias接近count的值。
