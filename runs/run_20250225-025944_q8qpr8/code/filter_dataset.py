'''
import pickle

# 读取 .pkl 文件
with open('/root/autodl-tmp/wangliang/weibo/hk/False_Business_0.pkl', 'rb') as file:
    data = pickle.load(file)

# 打印加载的数据
print(data)



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
