import pandas as pd

# 设置显示最大行数和列数
pd.set_option('display.max_rows', None)  # 不限制最大行数
pd.set_option('display.max_columns', None)  # 不限制最大列数
pd.set_option('display.width', None)  # 自适应列宽
pd.set_option('display.max_colwidth', None)  # 显示列的最大宽度

# 读取数据
#file_path = "/root/autodl-tmp/wangliang/weibo/hk/HK_weibo_all.tsv"
file_path ="/root/autodl-tmp/wangliang/weibo/hk/HK_100_user_profile.tsv"
with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
    df = pd.read_csv(file, sep='\t', on_bad_lines='skip')

# 显示前三行
print(df.head(1))
print("info:")
print(df.info())

# 查看数据的基本统计
print("describe:")
print(df.describe())
# 获取数值列的基本统计信息
numeric_columns = ['fans_number', 'weibo_number', 'friends_count', 'favourites_count', 'bi_followers_count']
print("numeric_columns:")
print(df[numeric_columns].describe())
# 查看不同用户类型的分布
print("df['user_type'].value_counts()")
print(df['user_type'].value_counts())

# 查看不同性别的分布
print("gender")
print(df['gender'].value_counts())

# 查看不同地区（省市）的分布
print("province")
print(df['province'].value_counts())
print("city")
print(df['city'].value_counts())



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
#https://i0.wp.com/+图片链接，不然会403

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


'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19951430 entries, 0 to 19951429
Data columns (total 30 columns):
 #   Column           Dtype  
---  ------           -----  
 0   @timestamp       float64
 1   content          float64
 2   crawler_time     object 
 3   device           object 
 4   id               object 
 5   nickname         object 
 6   pic_content      object 
 7   ping             object 
 8   publish_time     object 
 9   r_device         object 
 10  r_nickname       object 
 11  r_ping           object 
 12  r_time           object 
 13  r_url            object 
 14  r_user_id        object 
 15  r_user_type      object 
 16  r_weibo_content  object 
 17  r_weibo_id       object 
 18  r_zan            object 
 19  r_zhuan          object 
 20  retweet          object 
 21  time             object 
 22  touxiang         object 
 23  url              object 
 24  user_id          object 
 25  user_type        object 
 26  weibo_content    object 
 27  weibo_id         object 
 28  zan              object 
 29  zhuan            object 
dtypes: float64(2), object(28)
memory usage: 4.5+ GB
None
       @timestamp       content
count         0.0  3.000000e+00
mean          NaN  2.929503e+15
std           NaN  2.537048e+15
min           NaN  0.000000e+00
25%           NaN  2.191607e+15
50%           NaN  4.383215e+15
75%           NaN  4.394255e+15
max           NaN  4.405294e+15
'''

'''
info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3124603 entries, 0 to 3124602
Data columns (total 25 columns):
 #   Column              Dtype  
---  ------              -----  
 0   id                  float64
 1   crawler_date        object 
 2   crawler_time        object 
 3   user_id             object 
 4   nickname            object 
 5   touxiang            object 
 6   user_type           object 
 7   gender              object 
 8   verified_type       object 
 9   verified_reason     object 
 10  description         object 
 11  fans_number         object 
 12  weibo_number        object 
 13  type                object 
 14  friends_count       object 
 15  favourites_count    object 
 16  created_at          object 
 17  allow_all_comment   object 
 18  bi_followers_count  object 
 19  location            object 
 20  province            object 
 21  city                object 
 22  domain              object 
 23  status              object 
 24  insert_time         object 
dtypes: float64(1), object(24)
memory usage: 596.0+ MB
None



describe:
                 id
count  3.124602e+06
mean   8.325697e+08
std    2.476303e+08
min    7.100000e+01
25%    7.619281e+08
50%    9.659032e+08
75%    1.000529e+09
max    1.008392e+09



numeric_columns:
        fans_number  weibo_number friends_count favourites_count bi_followers_count
count       3124584       3124583       3124577          3124574            3124566
unique        70484         70893          5955            16957               3536
top               1             2            \N               \N                 \N
freq          17515          9750        614707           614708             614707


df['user_type'].value_counts()
user_type
普通用户                             2605967
达人                                359515
黄V                                112836
蓝V                                 32438
金V                                  7048
微博女郎                                6752
\N                                     8
true                                   4
t                                      2
1                                      2
f                                      2
28                                     1
328                                    1
Have courage,and be kind.              1
Peace yo                               1
2019-11-04 18:07:53.965984+08          1
2011-08-08 00:39:22                    1
639                                    1
2019-11-05 08:03:27.263723+08          1
普                                      1
65                                     1
1553                                   1
2014-01-16 19:48:22                    1
m                                      1
false                                  1
348                                    1
5855                                   1
2010-12-04 18:21:08                    1
400                                    1
229                                    1
Name: count, dtype: int64


gender
gender
f                                                                           2158779
m                                                                            965767
\N                                                                               10
普通用户                                                                              2
false                                                                             2
541                                                                               1
也许它只是一种避世的方式 一种极致的放肆忘记虚伪与矜持 人生太累了太燥了才会想清火 你说生命只有一次 我当做只有一日 也不想麻木又无奈的流于形式          1
606                                                                               1
2019-11-05 03:24:23.68765+08                                                      1
关注我想关注的人，了解我想了解的世界，做人三观要正些才好。                                                     1
285                                                                               1
蒼井翔太🌸宅家的咸鱼文字工作者🌸ツキウサ最萌了                                                           1
其他                                                                                1
65                                                                                1
39                                                                                1
21                                                                                1
115                                                                               1
36                                                                                1
32                                                                                1
931                                                                               1
231                                                                               1
156                                                                               1
724                                                                               1
2019-11-05 08:21:48.740217+08                                                     1
“那个男孩躺在空无一人的房间里，闭上眼，等着天使来亲吻他的嘴唇。”                                                 1
“是如此漫长的雨季”                                                                        1
EXO ♡and NCT♡WANNA ONE                                                            1
886                                                                               1
153                                                                               1
177                                                                               1
1                                                                                 1
true                                                                              1
1000                                                                              1
106                                                                               1
中高级达人                                                                             1



Name: count, dtype: int64
province
province
\N                               614704
100                              312139
44                               286447
11                               178761
400                              165034
32                               148520
33                               129373
31                               127129
37                               117180
51                               103847
41                                79658
35                                77459
42                                74769
43                                63937
13                                59839
21                                59129
34                                56806
61                                52577
50                                48890
45                                47656
36                                37863
14                                36848
12                                35900
23                                33155
53                                27994
22                                25869
15                                20114
81                                20003
52                                19265
62                                14365
46                                13287
65                                12834
71                                 6978
64                                 6108
82                                 3524
63                                 3422
54                                 3162
t                                     7
2019-11-05 08:21:48.740217+08         2
0                                     2
2019-11-04 19:00:57.222437+08         1
2019-11-05 08:35:04.620495+08         1
1000                                  1
Name: count, dtype: int64



city
city
1000                             741376
1                                640232
\N                               614702
2                                148535
3                                138506
5                                129678
6                                 87894
8                                 77848
7                                 76611
4                                 74420
10                                47265
15                                42351
9                                 41466
16                                40110
13                                33529
12                                31849
11                                30278
14                                17813
19                                15876
17                                13083
18                                10388
20                                10009
22                                 6213
23                                 4850
26                                 4740
28                                 4186
52                                 4162
29                                 4019
51                                 3297
24                                 2917
25                                 2787
27                                 2777
21                                 2543
31                                 2279
90                                 2170
32                                 1822
53                                 1498
34                                 1467
30                                 1286
40                                 1071
35                                  734
33                                  657
83                                  613
82                                  605
43                                  550
42                                  544
45                                  454
81                                  450
44                                  390
36                                  300
39                                  283
38                                  257
84                                  248
37                                  247
41                                  154
46                                  154
t                                     3
0                                     2
2019-11-05 08:24:27.770841+08         2
2019-11-05 08:32:25.705902+08         1
2019-11-05 08:19:10.488622+08         1
2019-11-05 08:37:43.426445+08         1
2019-11-05 06:15:45.800483+08         1
2019-11-04 13:52:00.095556+08         1
Name: count, dtype: int64
'''