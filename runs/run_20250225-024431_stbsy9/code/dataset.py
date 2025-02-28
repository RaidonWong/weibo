import pandas as pd

# 设置显示最大行数和列数
pd.set_option('display.max_rows', None)  # 不限制最大行数
pd.set_option('display.max_columns', None)  # 不限制最大列数
pd.set_option('display.width', None)  # 自适应列宽
pd.set_option('display.max_colwidth', None)  # 显示列的最大宽度

# 读取数据
#file_path = "/root/autodl-tmp/wangliang/weibo/hk/HK_weibo_all.tsv"
#file_path ="/root/autodl-tmp/wangliang/weibo/hk/HK_100_user_profile.tsv"
#file_path ="/root/autodl-tmp/wangliang/weibo/hk/HK_weibo_0811.tsv"
#with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
    #df = pd.read_csv(file, sep='\t', on_bad_lines='skip')


import pandas as pd

# 1. 加载 .tsv 文件
df = pd.read_csv('\\root\\autodl-tmp\\wangliang\\weibo\\hk\\HK_weibo_0811.tsv', sep='\t', header=None)

# 假设微博正文在第6列（根据你提供的数据示例调整列的索引）
# 你可以查看 df.columns 来了解每一列的具体含义
weibo_text_column = 13  # 微博正文列索引（从0开始）

# 2. 筛选包含“香港机场”相关的内容
keywords = ['香港机场', '香港', '机场','航班']  # 你可以根据需要添加更多关键词

# 3. 定义一个函数来检查微博正文是否包含任何关键词
def contains_keywords(text):
    return any(keyword in text for keyword in keywords)

# 4. 应用筛选函数，创建一个新的DataFrame，仅包含相关内容
filtered_df = df[df[weibo_text_column].apply(lambda x: contains_keywords(str(x)))]

# 5. 查看筛选后的数据
print(filtered_df)

# 可选：将筛选后的数据保存为新的 TSV 文件
filtered_df.to_csv('/root/autodl-tmp/wangliang/weibo/hk/filtered_weibo.tsv', sep='\t', index=False)

















'''
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
# https://i0.wp.com/+图片链接，不然会403

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



"""
2019-11-06T23:42:45	微博 HTML5 版	38873694220	round_theworld2	d827c8d6ly1g8op1hgfwsj20q81j2wk1,d827c8d6ly1g8op1h11l2j20q91kw0xh,d827c8d6ly1g8op8otro5j20vc2zxx1o	0	2019-11-06T22:25:13	HUAWEI Mate30 Pro 5G	HW前HR	71	2019-11-06T14:44:14	weibo.com/3626485974/If0k5tl8T	3626485974	金V	美国Facebook太可怕了，在何君尧被刺案中，居然有人通过后台恶意修改新闻时间造假来支持香港暴徒！\n图1，事件明明发生在今天，文汇报在医院的采访也是在今天。\n然而图2，Facebook有人故意在后台把文汇报的视频时间改为昨天晚上7点多，从而在废青暴徒中宣传这是一次演戏。\n说是黑客，但都知道是Facebook ​​​	4435761656992787	376	93	True		https://tvax1.sinaimg.cn/crop.0.0.996.996.50/006z8ElNly8fv7im57xbjj30ro0rojtd.jpg?KID=imgbed,tva&Expires=1573094564&ssig=Ns4h3zrfm%2B	weibo.com/6016030487/If3ld652C	6016030487	普通用户	转发微博	4435877671449350	0	0
2019-11-24T04:35:22	iPhone客户端	39626843018	1识少少扮代表1	None	0	2019-11-23T16:37:17	None	曝料一哥	205	2019-11-23T15:57:04	weibo.com/6011391162/IhB4603pl	6011391162	金V	林太全身都在拒绝与李握手！好样的，挺你[good][good][good] #香港被砸身亡罗伯儿子发声# http://t.cn/AidTtpxP  #烧国旗13岁少女认罪悔过# ​​​ http://f.video.weibocdn.com/000QIO4blx07yOmMkzRC0104120039bn0E010.mp4?label=mp4_ld&template=640x360.25.0&trans_finger=40a32e8439c5409a63ccf853562a60ef&Expires=1574573688&ssig=MWevUl1VQy&KID=unistore,video	4441940580013103	2497	216	True		https://tvax4.sinaimg.cn/crop.0.0.1080.1080.50/00713tUkly8g3z8v4yis8j30u00u0q5a.jpg?KID=imgbed,tva&Expires=1574580920&ssig=cHxc9EUxa4	weibo.com/6428536112/IhBkq1u5x	6428536112	普通用户	[doge]	4441950700353991	0	0
2019-11-14T06:38:51	小米8青春版 潮流旗舰	39198927783	775看天下	None	0	2019-11-14T03:07:44	西安直播超话	西安直播	46	2019-11-14T02:25:39	weibo.com/2268603763/Ig8UgDBmy	2268603763	金V	#西安爆料#【善良女子下班遇“香港”豪车男借钱，#转账2500元后追悔莫及#】近日，西安一女孩在路上走着，突然被一白色奥迪车的驾驶人员叫住，对方南方口音自称是香港人，称他在高速服务区去上卫生间，公文包在副驾驶被偷了。经过一系列骗术，让女孩给她取了2500元现金。http://t.cn/AiradVG2 ​​​ http://f.video.weibocdn.com/001sEk3ylx07yyI8hvqU01041200eDUk0E010.mp4?label=mp4_ld&template=640x360.25.0&trans_finger=bdef57f06ae52835a2c783ca389e8517&Expires=1573717129&ssig=SHIshafmmb&KID=unistore,video	4438474889438418	29	10	True		https://tva1.sinaimg.cn/crop.0.0.720.720.50/a4e5d6e7jw8ez46mqqx5kj20k00k03zq.jpg?KID=imgbed,tva&Expires=1573724329&ssig=qfzEPDrRyU	weibo.com/2766526183/Ig9blFlkl	2766526183	普通用户	转发微博	4438485479853433	0	0
2019-11-18T23:39:11	iPhone客户端	39404222052	loeywoon	a716fd45ly1g91d08wb1kj20fo0lx0y5	0	2019-11-18T05:08:48	微博 weibo.com	人民日报	11503	2019-11-17T13:41:07	weibo.com/2803301701/IgFBWeW73	2803301701	蓝V	【#救救李伯#】他没有阻碍行人，没有破坏一砖一瓦，只因意见不同，就被暴徒纵火烧身[伤心]#香港被烧老伯随时有生命危险# ​​​	4439732043559981	228465	20780	True		https://tvax4.sinaimg.cn/crop.0.0.1080.1080.50/6d21adfaly8g0hefs2ht6j20u00u0tb8.jpg?KID=imgbed,tva&Expires=1574131145&ssig=nVTGyxEF4M	weibo.com/1830923770/IgLGtCh4c	1830923770	普通用户	//@还是就叫苏亦遥吧:李伯我们能救，但香港只能自救	4439965499122072	0	0
2019-11-11T19:39:08	iPhone客户端	39090435174	一闪一闪亮晶晶圣所里的小星星	6cfd34c7gy1g8u2x9akjdj218g0qe48a	0	2019-11-11T13:15:37	林俊杰超话	李皎皎李	150	2019-11-11T06:26:16	weibo.com/1828533447/IfIcrAfgZ	1828533447	黄V	#林俊杰[超话]# \n啊啊啊啊啊啊啊[泪]\n香港等你@林俊杰 ​	4437448278638521	523	151	True		https://tvax2.sinaimg.cn/crop.0.0.512.512.50/9b120c97ly8g8s8qarcuqj20e80e874p.jpg?KID=imgbed,tva&Expires=1573489696&ssig=2vKawZVXZm	weibo.com/2601651351/IfKSBjYWD	2601651351	普通用户	？？？？？？？？？？是有毛病吗？不看新闻的吗[费解]	4437551294762507	0	0
2019-11-19T11:27:52	荣耀9 美得有声有色	39425032981	冰灵风de世界	None	0	2019-11-19T00:53:12	iPhone 7 Plus	观察者网	801	2019-11-18T15:39:59	weibo.com/1887344341/IgPOGph9S	1887344341	蓝V	【酒精烧伤严重不严重？有人试了一把[费解]】在此前香港暴徒在街头纵火谋杀一名老人的事情发生后，有暴徒支持者称被烧老人伤势不重、酒精烧人伤害不大，曾多次在自己的境外社交账号上谴责警方暴力的内地网络博主陈秋实，在被人问到如何看待此事后拒绝表态谴责，随后发布了一段他在自己的手臂上撒上酒精 ​​​ http://miaopai.video.weibocdn.com/004flBXJlx07yGUYMVjq01041200h5JD0E013.mp4?label=mp4_ld&template=640x360.25.0&trans_finger=40a32e8439c5409a63ccf853562a60ef&Expires=1574166468&ssig=Nfq1%2Bx7EVT&KID=unistore,video	4440124346024160	7855	832	True		https://tva1.sinaimg.cn/crop.0.0.180.180.50/7090b90bjw1e8qgp5bmzyj2050050aa8.jpg?KID=imgbed,tva&Expires=1574173668&ssig=P7Ecq6uQLR	weibo.com/1888532747/IgTreuCa8	1888532747	普通用户	这货就是个脑残，连最基本的常识都没有，还经常挂个民主人士的头衔给人大讲道理，制造假新闻来支撑自己的理论，每次被啪啪打脸都嘴硬的来一句“房贷还了吗，存款过六位数了吗，少关心那没用的”搞的人一脸“？？？？？”[二哈]	4440263567296540	0	0
2019-11-10T13:35:53	360安全浏览器	39031566692	天涯叟客	None	0	2019-11-10T12:05:19	祖国反黑站超话	港闻速递	349	2019-11-10T09:53:11	weibo.com/2708476577/IfA7WcOjW	2708476577	金V	香港上水，多名黑衣暴徒追打一名内地游客，男游客被迫扔掉行李箱逃跑！暴徒越发猖狂，大白天就已经抢劫，如果没有必要就别来了，拖着行李箱太显眼了！遇到危险，别管箱子，直接跑！\n\n#祖国反黑站[超话]#@海港SirHabaSir @孤烟暮蝉 @香港唐僧阿Sir林景昇 @KingKingHKG @大健聰 @香港光頭警長  ​​​ http://f.video.weibocdn.com/003X5LXngx07ytNniHKf010412001QUr0E010.mp4?label=mp4_ld&template=360x632.24.0&trans_finger=81b11b8c5ffb62d33ceb3244bdd17e7b&Expires=1573396523&ssig=8EdmvfYC3O&KID=unistore,video	4437137963053372	1677	500	True		https://tva3.sinaimg.cn/crop.9.12.168.168.50/4c90d7fajw8fbfot29cukj2050050aa7.jpg?KID=imgbed,tva&Expires=1573403751&ssig=isGV9HVMk4	weibo.com/1284560890/IfAZzEzIX	1284560890	达人	[怒][怒][怒]//@凡宝移山: //@凯雷:这TM的公开抢劫内地游客......#香港、香港# #香港局势# //@風中微塵:搶劫！//@居里小阿姨:这是抢劫吧？	4437171219670447	0	0

"""

"""https://weibo.com/6016030487/If3ld652C"""
"""https://weibo.com/3626485974/If0k5tl8T"""


'''
2019-11-06T23:42:45	
微博 HTML5 版	                device
38873694220	                    id                 https://weibo.com/u/3626485974       https://weibo.com/u/+id
round_theworld2	                nickname
d827c8d6ly1g8op1hgfwsj20q81j2wk1,d827c8d6ly1g8op1h11l2j20q91kw0xh,d827c8d6ly1g8op8otro5j20vc2zxx1o	picture
0	                            ping
2019-11-06T22:25:13 publish time
HUAWEI Mate30 Pro 5G	r_device
HW前HR	                r_nickname
71	                    r_ping
2019-11-06T14:44:14	    r_time  
weibo.com/3626485974/If0k5tl8T	    r_url
3626485974	                r_user_id
金V	                        r_user_type
美国Facebook太可怕了，在何君尧被刺案中，居然有人通过后台恶意修改新闻时间造假来支持香港暴徒！\n图1，事件明明发生在今天，文汇报在医院的采访也是在今天。\n然而图2，Facebook有人故意在后台把文汇报的视频时间改为昨天晚上7点多，从而在废青暴徒中宣传这是一次演戏。\n说是黑客，但都知道是Facebook ​​​	
4435761656992787	        r_weibo_id
376	                        r_zan
93	                        r_zhuan
True		                retweet
https://tvax1.sinaimg.cn/crop.0.0.996.996.50/006z8ElNly8fv7im57xbjj30ro0rojtd.jpg?KID=imgbed,tva&Expires=1573094564&ssig=Ns4h3zrfm%2B	touxiang
weibo.com/6016030487/If3ld652C	  url
6016030487	                user_id
普通用户	                    user_type
转发微博	                    weibo_content
4435877671449350	            weibo_id
0	                            zan
0                           zhuan

data = {
    "@timestamp": None,
    "content": None,
    "crawler_time": "2019-08-15T18:28:10+08:00",
    "device": "iPhone客户端",
    "id": 35711584708,
    "nickname": "我不以为什么",
    "pic_content": "884f7263gy1g5z6mzhichj20j90edmxj,884f7263gy1g5z6n4ktq9j20dy09g3zt",##https://i0.wp.com/tvax1.sinaimg.cn/crop.0.0.996.996.50/+这串.jpg可访问
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