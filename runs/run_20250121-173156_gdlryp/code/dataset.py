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


data = {
    "@timestamp": None,
    "content": None,
    "crawler_time": "2019-08-15T18:28:10+08:00",
    "device": "iPhone客户端",
    "id": 35711584708,
    "nickname": "我不以为什么",
    "pic_content": "884f7263gy1g5z6mzhichj20j90edmxj,884f7263gy1g5z6n4ktq9j20dy09g3zt",
    "ping": 0,
    "publish_time": "2019-08-15T18:06:34+08:00",
    "r_device": "专业版微博",
    "r_nickname": "人民网",
    "r_ping": 23615,
    "r_time": "2019-08-14T14:20:15+08:00",
    "r_url": "weibo.com/2286908003/I2b7OiShO",
    "r_user_id": 2286908003,
    "r_user_type": "蓝V",
    "r_weibo_content": "#香港是中国的香港#，香港事务纯属中国内政，不容任何国家、组织或个人以任何方式干预。反对暴力，严惩暴徒，守护法治，珍惜安宁！ ​​​",
    "r_weibo_id": 4405194244498584.0,
    "r_zan": 307004.0,
    "r_zhuan": 155269.0,
    "retweet": True,
    "time": None,
    "touxiang": "https://tva3.sinaimg.cn/crop.0.0.511.511.50/68e9de8fjw8eyqh8k3yq6j20e70e8mxc.jpg?KID=imgbed,tva&Expires=1565875689&ssig=cGOLowS2cJ",
    "url": "weibo.com/1760157327/I2m2b63Wp",
    "user_id": 1760157327.0,
    "user_type": "普通用户",
    "weibo_content": "转发微博",
    "weibo_id": 4405613591445121.0,
    "zan": 0.0,
    "zhuan": 0.0
}

