from collections import Counter
import os

import numpy as np
import pandas as pd


def pad_1d(array, max_len):  # 对于max_len，返回长度=max_len的，其中如果长度不足则补零，超过则截断
    array = array[:max_len]
    length = len(array)
    padded = array + [0]*(max_len - len(array))
    return padded, length


def make_word_idx(product_names):
    words = [word for name in product_names for word in name.split()]  # 把商品名拆分成单词
    word_counts = Counter(words)  # 构建word: 计数

    max_id = 1
    word_idx = {}
    for word, count in word_counts.items():
        if count < 10:
            word_idx[word] = 0
        else:
            word_idx[word] = max_id
            max_id += 1

    return word_idx  # 构建word -> idx的字典，其中计数<10的idx为0，计数>=10的word的idx从1开始递增


def encode_text(text, word_idx):
    return ' '.join([str(word_idx[i]) for i in text.split()]) if text else '0'


if __name__ == '__main__':
    product_data = pd.read_csv('../../data/processed/product_data.csv')  # 按照每行用户的数据展开，包含用户最后一个订单之前的所有订单中的商品set情况
    product_data['product_name'] = product_data['product_name'].map(lambda x: x.lower())

    product_df = pd.read_csv('../../data/raw/products.csv')  # product_id(产品唯一ID),product_name(商品名称),aisle_id,department_id
    product_df['product_name'] = product_df['product_name'].map(lambda x: x.lower())

    word_idx = make_word_idx(product_df['product_name'].tolist())
    product_data['product_name_encoded'] = product_data['product_name'].map(lambda x: encode_text(x, word_idx))

    num_rows = len(product_data)  # （每个用户之前订单的商品ID的set然后+1） * 用户个数

    user_id = np.zeros(shape=[num_rows], dtype=np.int32)
    product_id = np.zeros(shape=[num_rows], dtype=np.int32)
    aisle_id = np.zeros(shape=[num_rows], dtype=np.int16)
    department_id = np.zeros(shape=[num_rows], dtype=np.int8)
    eval_set = np.zeros(shape=[num_rows], dtype='S5')
    label = np.zeros(shape=[num_rows], dtype=np.int8)

    is_ordered_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    index_in_order_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_dow_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_hour_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    days_since_prior_order_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_size_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    reorder_size_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    order_number_history = np.zeros(shape=[num_rows, 100], dtype=np.int8)
    product_name = np.zeros(shape=[num_rows, 30], dtype=np.int32)
    product_name_length = np.zeros(shape=[num_rows], dtype=np.int8)
    history_length = np.zeros(shape=[num_rows], dtype=np.int8)

    for i, row in product_data.iterrows():
        if i % 10000 == 0:
            print(i, num_rows)

        user_id[i] = row['user_id']
        product_id[i] = row['product_id']
        aisle_id[i] = row['aisle_id']
        department_id[i] = row['department_id']
        eval_set[i] = row['eval_set']
        label[i] = row['label']

        is_ordered_history[i, :], history_length[i] = pad_1d(map(int, row['is_ordered_history'].split()), 100)
        index_in_order_history[i, :], _ = pad_1d(map(int, row['index_in_order_history'].split()), 100)
        order_dow_history[i, :], _ = pad_1d(map(int, row['order_dow_history'].split()), 100)
        order_hour_history[i, :], _ = pad_1d(map(int, row['order_hour_history'].split()), 100)
        days_since_prior_order_history[i, :], _ = pad_1d(map(int, row['days_since_prior_order_history'].split()), 100)
        order_size_history[i, :], _ = pad_1d(map(int, row['order_size_history'].split()), 100)
        reorder_size_history[i, :], _ = pad_1d(map(int, row['reorder_size_history'].split()), 100)
        order_number_history[i, :], _ = pad_1d(map(int, row['order_number_history'].split()), 100)
        product_name[i, :], product_name_length[i] = pad_1d(map(int, row['product_name_encoded'].split()), 30)

    if not os.path.isdir('data'):
        os.makedirs('data')

    np.save('data/user_id.npy', user_id)
    np.save('data/product_id.npy', product_id)
    np.save('data/aisle_id.npy', aisle_id)
    np.save('data/department_id.npy', department_id)
    np.save('data/eval_set.npy', eval_set)
    np.save('data/label.npy', label)

    np.save('data/is_ordered_history.npy', is_ordered_history)
    np.save('data/index_in_order_history.npy', index_in_order_history)
    np.save('data/order_dow_history.npy', order_dow_history)
    np.save('data/order_hour_history.npy', order_hour_history)
    np.save('data/days_since_prior_order_history.npy', days_since_prior_order_history)
    np.save('data/order_size_history.npy', order_size_history)
    np.save('data/reorder_size_history.npy', reorder_size_history)
    np.save('data/order_number_history.npy', order_number_history)
    np.save('data/product_name.npy', product_name)
    np.save('data/product_name_length.npy', product_name_length)
    np.save('data/history_length.npy', history_length)
