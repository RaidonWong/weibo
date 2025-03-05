'''
import pickle

# 读取 .pkl 文件
with open('/root/autodl-tmp/wangliang/weibo/hk/False_Business_0.pkl', 'rb') as file:
    data = pickle.load(file)

# 打印加载的数据
print(data)



'''
'''
import pandas as pd

# 使用 open() 以指定编码读取文件
with open('/root/autodl-tmp/wangliang/weibo/hk/HK_weibo_0811.tsv', 'r', encoding='utf-8', errors='ignore') as f:
    df = pd.read_csv(f, sep='\t', header=None, on_bad_lines='skip')

# 或者尝试 latin1
# df = pd.read_csv('/root/autodl-tmp/wangliang/weibo/hk/HK_weibo_0811.tsv', sep='\t', header=None, encoding='latin1')

# 假设微博正文在第6列（根据你提供的数据示例调整列的索引）
# 你可以查看 df.columns 来了解每一列的具体含义
weibo_text_column = 16  # 微博正文列索引（从0开始）
print(df[weibo_text_column].head())


# 2. 筛选包含“香港机场”相关的内容
keywords = ['政府','议员','议会','警']  # 你可以根据需要添加更多关键词

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


import pandas as pd
output_dir = "/root/autodl-tmp/wangliang/weibo/hk/"
with open("/root/autodl-tmp/wangliang/weibo/hk/demo100.tsv", 'r', encoding='utf-8', errors='ignore') as f:
    df = pd.read_csv(f, sep='\t', header=None, on_bad_lines='skip')


id=df.iloc[:, 3]
with open("/root/autodl-tmp/wangliang/weibo/hk/HK_weibo_all.tsv", 'r', encoding='utf-8', errors='ignore') as f:
    df1 = pd.read_csv(f, sep='\t', header=None, on_bad_lines='skip')

data=df1[df1.iloc[:, 24].isin(id)]
top_sorted = data.sort_values(by=data.columns[24], ascending=True)
top_sorted.to_csv(f"{output_dir}/filter_user_allcontent.tsv", index=False, sep="\t")
'''
# 读取数据
with open("/root/autodl-tmp/wangliang/weibo/hk/topevent.tsv", 'r', encoding='utf-8', errors='ignore') as f:
    df = pd.read_csv(f, sep='\t', header=None, on_bad_lines='skip')

event_id=df.iloc[:, 17]

with open("/root/autodl-tmp/wangliang/weibo/hk/HK_weibo_all.tsv", 'r', encoding='utf-8', errors='ignore') as f:
    df1 = pd.read_csv(f, sep='\t', header=None, on_bad_lines='skip')

r_user_id=df1[df1.iloc[:, 17].isin(event_id)]

with open("/root/autodl-tmp/wangliang/weibo/hk/HK_100_user_profile.tsv", 'r', encoding='utf-8', errors='ignore') as f:
    df2 = pd.read_csv(f, sep='\t', header=None, on_bad_lines='skip')

r_user=df2[df2.iloc[:, 3].isin(r_user_id.iloc[:, 24])]

r_user_top=r_user.head(150)
top=df2[df2.iloc[:, 3].isin(r_user_top.iloc[:, 3])]
top_sorted = top.sort_values(by=top.columns[3], ascending=True)
top_sorted.to_csv(f"{output_dir}/demo100.tsv", index=False, sep="\t")
'''
'''
event_id_counts = df.iloc[:, 17].value_counts()  # 使用列索引24来统计user_id
top_event_id = event_id_counts.head(100)
top_event_data = df[df.iloc[:, 17].isin(top_event_id.index)] 
top_event_data.iloc[:, 17] = pd.to_numeric(top_event_data.iloc[:, 17], errors='coerce')
top_event_data_sorted = top_event_data.sort_values(by=top_event_data.columns[17], ascending=True)

top_user_data_unique = top_event_data_sorted.drop_duplicates(subset=top_event_data_sorted.columns[17], keep='first')
# 确保列是数值类型


top_user_data_unique.to_csv(f"{output_dir}/topevent.tsv", index=False, sep="\t")
'''
'''
filtered_data = df[df.iloc[:, 3] == 2803301701.0]

# 查看筛选后的数据
output_dir = "/root/autodl-tmp/wangliang/weibo/hk/"
filtered_data.to_csv(f"{output_dir}/renmin.tsv", index=False, sep="\t")
'''

'''

user_id_counts = df.iloc[:, 24].value_counts()  # 使用列索引24来统计user_id
#r_user_id_counts = df.iloc[:, 14].value_counts()  # 使用列索引14来统计r_user_id
#r_weibo_id_counts = df.iloc[:, 17].value_counts()  # 使用列索引17来统计r_weibo_id

# 获取出现次数最多的 user_id 和 r_user_id
top_user_id = user_id_counts.head(10)
#top_r_user_id = r_user_id_counts.head(10)

print(top_user_id)
#top_user_data = df[df.iloc[:, 24].isin(top_user_id.index)]  # 使用列索引24来筛选user_id
#top_user_data_sorted = top_user_data.sort_values(by=top_user_data.columns[24], ascending=True)

import os
'''
'''
# 确保目录存在
output_dir = "/root/autodl-tmp/wangliang/weibo/hk/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存文件到指定目录
top_user_data_sorted.to_csv(f"{output_dir}/alltop10userdata.tsv", index=False, sep="\t")


print("Data has been saved to 'user_r_user_data_summary.tsv'")
# 筛选出相关的整条数据
'''
'''
top_user_data = df[df.iloc[:, 24].isin(top_user_id.index)]  # 使用列索引24来筛选user_id
top_r_user_data = df[df.iloc[:, 14].isin(top_r_user_id.index)]  # 使用列索引14来筛选r_user_id

# 合并这两个DataFrame
result_data = pd.concat([top_user_data, top_r_user_data])


# 将结果保存到新的 .tsv 文件中
import os

# 确保目录存在
output_dir = "/root/autodl-tmp/wangliang/weibo/hk/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存文件到指定目录
result_data.to_csv(f"{output_dir}/user_r_user_data_summary.tsv", index=False, sep="\t")


print("Data has been saved to 'user_r_user_data_summary.tsv'")
'''