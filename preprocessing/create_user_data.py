import os

import pandas as pd


def parse_order(x):
    series = pd.Series()

    series['products'] = '_'.join(x['product_id'].values.astype(str).tolist())
    series['reorders'] = '_'.join(x['reordered'].values.astype(str).tolist())
    series['aisles'] = '_'.join(x['aisle_id'].values.astype(str).tolist())
    series['departments'] = '_'.join(x['department_id'].values.astype(str).tolist())

    series['order_number'] = x['order_number'].iloc[0]
    series['order_dow'] = x['order_dow'].iloc[0]
    series['order_hour'] = x['order_hour_of_day'].iloc[0]
    series['days_since_prior_order'] = x['days_since_prior_order'].iloc[0]

    return series


def parse_user(x):
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

    series['eval_set'] = x['eval_set'].values[-1]

    return series

if __name__ == '__main__':
    # order_id(订单唯一ID),user_id(用户唯一ID),eval_set(属于prior/train/test),order_number(在这个用户中下单的顺序),order_dow(星期几),order_hour_of_day,days_since_prior_order
    # ID1, User1, prior, 1, 2, 08,
    # ID2, User1, prior, 2, 3, 07, 15.0
    # ID3, User1, train, 3, 3, 12, 21.0
    orders = pd.read_csv('../data/raw/orders.csv')

    # order_id(订单唯一ID),product_id(产品唯一ID),add_to_cart_order(加入购物车的顺序),reordered(0/1)
    # 1, 49302, 1, 1
    # 1, 11109, 2, 1
    # 1, 10246, 3, 0
    prior_products = pd.read_csv('../data/raw/order_products__prior.csv')
    train_products = pd.read_csv('../data/raw/order_products__train.csv')
    order_products = pd.concat([prior_products, train_products], axis=0)  # 简单按行合并prior和train

    # product_id(产品唯一ID),product_name(),aisle_id,department_id
    # 1, Chocolate Sandwich Cookies, 61, 19
    # 2, All - Seasons Salt, 104, 13
    # 3, Robust Golden Unsweetened Oolong Tea, 94, 7
    products = pd.read_csv('../data/raw/products.csv')

    df = orders.merge(order_products, how='left', on='order_id')
    df = df.merge(products, how='left', on='product_id')
    df['days_since_prior_order'] = df['days_since_prior_order'].fillna(0).astype(int)
    null_cols = ['product_id', 'aisle_id', 'department_id', 'add_to_cart_order', 'reordered']
    df[null_cols] = df[null_cols].fillna(0).astype(int)

    if not os.path.isdir('../data/processed'):
        os.makedirs('../data/processed')

    user_data = df.groupby('user_id', sort=False).apply(parse_user).reset_index()
    user_data.to_csv('../data/processed/user_data.csv', index=False)
