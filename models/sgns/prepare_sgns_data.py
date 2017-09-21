import os

import numpy as np
import pandas as pd


if __name__ == '__main__':
    # 每个user_id为一行，后面的列为这个用户的订单信息以及各个订单购买的商品信息等等
    user_data = pd.read_csv('../../data/processed/user_data.csv')

    x = []
    y = []
    for _, row in user_data.iterrows():
        if _ % 10000 == 0:
            print(_)

        user_id = row['user_id']  # 每一row只有一个user_id
        products = row['product_ids']  # 每一row的product_ids由多个“id1_id2 id1_id2 id1_id2_id3”组成
        products = ' '.join(products.split()[:-1])  # 排除最后一个订单的products，组成一个长链
        for order in products.split():  # 获得每个订单的商品IDs
            items = order.split('_')  # 获得一个订单的商品list
            for i in range(len(items)):  # 假设这时有个items里面有10个商品，那么有：
                                        # i=0, j in (0, 3)
                                        # i=1, j in (0, 4)
                                        # i=2, j in (0, 5)
                                        # i=3, j in (1, 6)
                                        # ...  ...
                                        # i=6, j in (4, 9)
                                        # i=7, j in (5, 10)
                                        # i=8, j in (6, 10)
                                        # i=9, j in (7, 10)
                for j in range(max(0, i - 2), min(i + 3, len(items))):
                    if i != j:
                        x.append(int(items[j]))  # 1 2 0 2 3 0 1 3 4 ...
                        y.append(int(items[i]))  # 0 0 1 1 1 2 2 2 2 ...

    # 对于一次订单中的每个商品，从0到最后一个，假设当前是第i个：（下面的i-2不能<0，i+3不能>这个订单中商品的总数）
    x = np.array(x)  # x是添加当前(i-2, i+3),且不是i的最多4个商品（有可能为0，当订单只有一个商品的时候）
    y = np.array(y)  # y是添加len((i-2, i+3))-1个第i号商品，最多4个

    if not os.path.isdir('data'):
        os.makedirs('data')

    np.save('data/x.npy', x)
    np.save('data/y.npy', y)
    """
    最后得到了在所有用户的每个订单中，商品ID之间的局部相互关系的数据x和y
    """
