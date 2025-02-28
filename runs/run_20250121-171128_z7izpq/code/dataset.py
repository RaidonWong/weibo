import pandas as pd

# 设置显示最大行数和列数
pd.set_option('display.max_rows', None)  # 不限制最大行数
pd.set_option('display.max_columns', None)  # 不限制最大列数
pd.set_option('display.width', None)  # 自适应列宽
pd.set_option('display.max_colwidth', None)  # 显示列的最大宽度

# 读取数据
file_path = "/root/autodl-tmp/wangliang/weibo/hk/HK_weibo_all.tsv"
with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
    df = pd.read_csv(file, sep='\t', on_bad_lines='skip')

# 显示前三行
print(df.head(1))

# 输出结果
user_data = {
    "id": 1.004432e+09,
    "crawler_date": "2018-01-28",
    "crawler_time": "2018-01-28 17:23:24+08",
    "user_id": 2203354057,
    "nickname": "孙坚强ooo",
    "touxiang": "https://tva2.sinaimg.cn/crop.0.0.1242.1242.50/835483c9jw8f24pachdy5j20yi0yiq5f.jpg",
    "user_type": "普通用户",
    "gender": "f",
    "verified_type": "普通用户",
    "verified_reason": "\\N",
    "description": "\\N",
    "fans_number": 313,
    "weibo_number": 123,
    "type": 1,
    "friends_count": 685,
    "favourites_count": 4,
    "created_at": "2011-06-27 21:31:31",
    "allow_all_comment": "true",
    "bi_followers_count": 30,
    "location": "河南 南阳",
    "province": 41,
    "city": 13,
    "domain": "\\N",
    "status": "t",
    "insert_time": "2019-11-05 08:29:45.561633+08"
}