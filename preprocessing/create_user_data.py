import os

import pandas as pd


def parse_order(x):  # 将一个用户的一次订单中的多个行记录，转成一行，其中商品信息改成字符串用"_"连接表达
    series = pd.Series()

    series['products'] = '_'.join(x['product_id'].values.astype(str).tolist())  # "productID1_productID2_productID3"
    series['reorders'] = '_'.join(x['reordered'].values.astype(str).tolist())
    series['aisles'] = '_'.join(x['aisle_id'].values.astype(str).tolist())
    series['departments'] = '_'.join(x['department_id'].values.astype(str).tolist())

    # 需要注意的是，丢弃了['eval_set']['add_to_cart_order']['product_name']的信息

    series['order_number'] = x['order_number'].iloc[0]
    series['order_dow'] = x['order_dow'].iloc[0]
    series['order_hour'] = x['order_hour_of_day'].iloc[0]
    series['days_since_prior_order'] = x['days_since_prior_order'].iloc[0]

    return series


def parse_user(x):  # x是按照user分组后的DataFrame，然后拍平成一行信息返回
    # 按照order_id分组，看各个订单的商品，并将订单中的各个商品信息合并在一个字符串中。
    parsed_orders = x.groupby('order_id', sort=False).apply(parse_order)

    series = pd.Series()

    series['order_ids'] = ' '.join(parsed_orders.index.map(str).tolist())
    series['order_numbers'] = ' '.join(parsed_orders['order_number'].map(str).tolist())
    series['order_dows'] = ' '.join(parsed_orders['order_dow'].map(str).tolist())
    series['order_hours'] = ' '.join(parsed_orders['order_hour'].map(str).tolist())
    series['days_since_prior_orders'] = ' '.join(parsed_orders['days_since_prior_order'].map(str).tolist())

    series['product_ids'] = ' '.join(parsed_orders['products'].values.astype(str).tolist())
    series['aisle_ids'] = ' '.join(parsed_orders['aisles'].values.astype(str).tolist())
    series['department_ids'] = ' '.join(parsed_orders['departments'].values.astype(str).tolist())
    series['reorders'] = ' '.join(parsed_orders['reorders'].values.astype(str).tolist())

    series['eval_set'] = x['eval_set'].values[-1]  # 返回train或者test

    return series

if __name__ == '__main__':
    # 侧重于用户的订单列表描述
    # order_id(订单唯一ID),user_id(用户唯一ID),eval_set(属于prior/train/test),order_number(在这个用户中下单的顺序),order_dow(星期几),order_hour_of_day,days_since_prior_order
    # ID1, User1, prior, 1, 2, 08,
    # ID2, User1, prior, 2, 3, 07, 15.0
    # ID3, User1, train, 3, 3, 12, 21.0
    orders = pd.read_csv('../data/raw/orders.csv')

    # 侧重于订单中的商品详情
    # order_id(订单唯一ID),product_id(产品唯一ID),add_to_cart_order(加入购物车的顺序),reordered(0/1)
    # 1, 49302, 1, 1
    # 1, 11109, 2, 1
    # 1, 10246, 3, 0
    prior_products = pd.read_csv('../data/raw/order_products__prior.csv')
    train_products = pd.read_csv('../data/raw/order_products__train.csv')
    order_products = pd.concat([prior_products, train_products], axis=0)  # 简单按行合并prior和train

    # 侧重于商品的描述
    # product_id(产品唯一ID),product_name(商品名称),aisle_id,department_id
    # 1, Chocolate Sandwich Cookies, 61, 19
    # 2, All - Seasons Salt, 104, 13
    # 3, Robust Golden Unsweetened Oolong Tea, 94, 7
    products = pd.read_csv('../data/raw/products.csv')

    # 按照orders中含有的order_id为行的选取标准，合并orders和order_products的列
    # order_id  user_id eval_set  order_number  order_dow  order_hour_of_day  days_since_prior_order  product_id  add_to_cart_order  reordered
    # 因为order_id在orders中只会出现一次，但是在order_products中出现多次，所以合并之后相当于把prior和train的数据添加了orders中的列内容。
    df = orders.merge(order_products, how='left', on='order_id')
    # 继续把products中的列product_name(),aisle_id,department_id加入进来
    df = df.merge(products, how='left', on='product_id')
    df['days_since_prior_order'] = df['days_since_prior_order'].fillna(0).astype(int)
    # 把NaN值填充为0
    null_cols = ['product_id', 'aisle_id', 'department_id', 'add_to_cart_order', 'reordered']
    df[null_cols] = df[null_cols].fillna(0).astype(int)

    if not os.path.isdir('../data/processed'):
        os.makedirs('../data/processed')

    # 按照user_id进行分组，看各个用户的订单
    user_data = df.groupby('user_id', sort=False).apply(parse_user).reset_index()
    user_data.to_csv('../data/processed/user_data.csv', index=False)

    # 最后输出每个user_id对应一行数据格式的csv
    # user_id  order_ids(1 2 3 ..)  order_numbers(1 2 3 ...)  order_dows(4 2 5 ...)  order_hours(09 23 15 ...)  days_since_prior_orders \
    # product_ids(id1_id2_id3 id1_id2_id3 ...) aisle_ids department_ids reorders eval_set(train/test)
