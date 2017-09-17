import os

import pandas as pd

if __name__ == '__main__':
    """
    基于商品的product_id构成一个含有过往信息的数据集
    """
    df = pd.read_csv('../data/processed/user_data.csv')

    products = pd.read_csv('../data/raw/products.csv')
    product_to_aisle = dict(zip(products['product_id'], products['aisle_id']))
    product_to_department = dict(zip(products['product_id'], products['department_id']))
    product_to_name = dict(zip(products['product_id'], products['product_name']))

    user_ids = []  # 最后一次订单之前的每个订单中商品ID的set，每个商品都有这个user_id
    product_ids = []  # 对于每个用户加入set中商品的ID，除了最后一个订单中的商品，而且尾部填充一个0
    aisle_ids = []  # 商品的aisle_id组成的list，除了最后一个订单中的商品，而且尾部填充一个0
    department_ids = []  # 商品的department_id组成的list，除了最后一个订单中的商品，而且尾部填充一个0
    product_names = []  # 商品的product_name组成的list，除了最后一个订单中的商品，而且尾部填充一个0
    eval_sets = []  # 用户的'eval_set' * 最后一单之前的商品个数 + 1个'eval_set'

    is_ordered_histories = []  # 记录这个商品在所有的过往购物订单中是否出现
    index_in_order_histories = []  # 记录这个商品在所有的过往购物订单中出现的次序（0表示没有出现）
    order_size_histories = []  # 记录过往订单中的商品个数
    reorder_size_histories = []  # 记录这个订单中在这个订单前出现的商品个数
    order_dow_histories = []  # 保持用户的原先订单中的'order_dows'信息
    order_hour_histories = []
    days_since_prior_order_histories = []
    order_number_histories = []

    labels = []  # 对于每个用户的每个product_set中的商品，如果用户是train类型，那么对于每个商品检测是否是最后一次购买的订单中（1 or 0），对于test类型，直接标记-1

    longest = 0
    for _, row in df.iterrows():  # row的内容有：user_id，空格连接的order_id，空格连接的order的顺序，空格连接的订单周几，空格连接的订单小时，空格连接的订单间隔，
        # 还有空格连接的商品s product_ids，aisle_ids department_ids reorders，以及eval_set
        if _ % 10000 == 0:
            print(_)

        user_id = row['user_id']
        eval_set = row['eval_set']

        products = row['product_ids']
        products, next_products = ' '.join(products.split()[:-1]), products.split()[-1]  # 分割最后一个订单和之前订单的商品

        reorders = row['reorders']
        reorders, next_reorders = ' '.join(reorders.split()[:-1]), reorders.split()[-1]  # 分割最后一个订单和之前订单的reorders

        product_set = set([int(j) for i in products.split() for j in i.split('_')])  # 获取除了最后一个订单之外所有订单的商品set集合
        next_product_set = set([int(i) for i in next_products.split('_')])  # 获取最后一个订单的商品set集合

        """
        注意，作者是用python2写的代码，在python2中，map直接产生list结果，而不是像在python3那样是个Iterator的返回
        """
        orders = [map(int, i.split('_')) for i in products.split()]  # 获取除了最后一个订单之外每个订单商品ID的map组成的list
        reorders = [map(int, i.split('_')) for i in reorders.split()]  # 获取除了最后一个订单之外所有订单中商品的reorders产生的map的list
        next_reorders = map(int, next_reorders.split('_'))  # 获取最后一个订单中商品的reorders产生的map

        for product_id in product_set:  # 对于这个用户除了最后一个订单外所有订单的商品ID组成的set遍历

            user_ids.append(user_id)  # 这个用户购买过多少商品就加入多少次这个用户的ID
            product_ids.append(product_id)  # 加入set中商品的ID
            labels.append(int(product_id in next_product_set) if eval_set == 'train' else -1)  # 如果当前用户被选为train，那么就把当前商品的ID是否在最后一次购买当中加入到label中，否则加入-1

            aisle_ids.append(product_to_aisle[product_id])  # 对于每个商品加入其aisle_id
            department_ids.append(product_to_department[product_id])  # 对于每个商品加入其department_id
            product_names.append(product_to_name[product_id])  # 对于每个商品加入其product_name
            eval_sets.append(eval_set)  # 对于每个商品加入用户的类型

            is_ordered = []  # 记录这个商品在所有的过往购物订单中是否出现
            index_in_order = []  # 记录这个商品在所有的过往购物订单中出现的次序（0表示没有出现）
            order_size = []  # 记录过往订单中的商品个数
            reorder_size = []  # 记录这个订单中在这个订单前出现的商品个数

            prior_products = set()  # 一边处理每个订单，一边把处理过的加入进去，形成与当前处理的订单相比的历史订单
            for order in orders:  # map的list，每个map对应一个订单的商品ID列表，从第一个订单直到倒数第二个订单
                is_ordered.append(str(int(product_id in order)))
                index_in_order.append(str(order.index(product_id) + 1) if product_id in order else '0')
                order_size.append(str(len(order)))
                reorder_size.append(str(len(prior_products & set(order))))
                prior_products |= set(order)

            is_ordered = ' '.join(is_ordered)
            index_in_order = ' '.join(index_in_order)
            order_size = ' '.join(order_size)
            reorder_size = ' '.join(reorder_size)

            is_ordered_histories.append(is_ordered)
            index_in_order_histories.append(index_in_order)
            order_size_histories.append(order_size)
            reorder_size_histories.append(reorder_size)
            order_dow_histories.append(row['order_dows'])
            order_hour_histories.append(row['order_hours'])
            days_since_prior_order_histories.append(row['days_since_prior_orders'])
            order_number_histories.append(row['order_numbers'])

        user_ids.append(user_id)
        product_ids.append(0)
        labels.append(int(max(next_reorders) == 0) if eval_set == 'train' else -1)

        aisle_ids.append(0)
        department_ids.append(0)
        product_names.append(0)
        eval_sets.append(eval_set)

        is_ordered = []
        index_in_order = []
        order_size = []
        reorder_size = []

        for reorder in reorders:
            is_ordered.append(str(int(max(reorder) == 0)))
            index_in_order.append(str(0))
            order_size.append(str(len(reorder)))
            reorder_size.append(str(sum(reorder)))

        is_ordered = ' '.join(is_ordered)
        index_in_order = ' '.join(index_in_order)
        order_size = ' '.join(order_size)
        reorder_size = ' '.join(reorder_size)

        is_ordered_histories.append(is_ordered)
        index_in_order_histories.append(index_in_order)
        order_size_histories.append(order_size)
        reorder_size_histories.append(reorder_size)
        order_dow_histories.append(row['order_dows'])
        order_hour_histories.append(row['order_hours'])
        days_since_prior_order_histories.append(row['days_since_prior_orders'])
        order_number_histories.append(row['order_numbers'])

    data = [
        user_ids,
        product_ids,
        aisle_ids,
        department_ids,
        product_names,
        is_ordered_histories,
        index_in_order_histories,
        order_size_histories,
        reorder_size_histories,
        order_dow_histories,
        order_hour_histories,
        days_since_prior_order_histories,
        order_number_histories,
        labels,
        eval_sets
    ]
    columns = [
        'user_id',
        'product_id',
        'aisle_id',
        'department_id',
        'product_name',
        'is_ordered_history',
        'index_in_order_history',
        'order_size_history',
        'reorder_size_history',
        'order_dow_history',
        'order_hour_history',
        'days_since_prior_order_history',
        'order_number_history',
        'label',
        'eval_set'
    ]
    if not os.path.isdir('../data/processed'):
        os.makedirs('../data/processed')

    df = pd.DataFrame(dict(zip(columns, data)))
    df.to_csv('../data/processed/product_data.csv', index=False)
