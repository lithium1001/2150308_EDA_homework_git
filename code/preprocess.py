import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import re
from datetime import datetime

# 读取Excel文件
excel_file = r"D:\personal\2024\py\eda\2150308_EDA_homework\data\phone_parameters_refined.xlsx"
df = pd.read_excel(excel_file)

# 品牌列表
brands = {
    'samsung': 'Samsung',
    'apple': 'Apple',
    'xiaomi': 'Xiaomi',
    'huawei': 'Huawei',
    'oppo': 'OPPO',
    'nokia': 'Nokia',
    'vivo': 'Vivo',
    'htc': 'HTC',
    'lenovo': 'Lenovo',
    'google': 'Google',
    'sony': 'Sony',
    'honor': 'Honor',
    'realme': 'Realme'
}

# 添加品牌列
# def get_brand(model):
#     if isinstance(model, str):
#         for brand, brand_name in brands.items():
#             if brand in model.lower():
#                 return brand_name
#     return 'Unknown'


# df['Brand'] = df['型号'].apply(get_brand)

# # 删除model中包含tablet的记录
# df = df[df['型号'].str.contains('Tablet', case=False)==False]
# df = df[df['型号'].str.contains('tablet', case=False)==False]

# df = df[df['型号'].str.contains('Tab', case=False)==False]

# 检查并移动数据
# def move_data(row):
#     if 'sim' in str(row['Body_Build']).lower():
#         row['Body_SIM'] = row['Body_Build']
#         row['Body_Build'] = ''
#     return row
# # 应用函数到DataFrame
# df = df.apply(move_data, axis=1)

# a
# # def move_data2(row):
# #     if pd.notnull(row['Body_Sim']):
# #         row['Body_SIM'] = row['Body_Sim']
# #         row['Body_Sim'] = ''
# #     return row
# a
# # 应用函数到DataFrame
# # df = df.apply(move_data2, axis=1)


# 提取bodyweight中的重量数据，单位g
# def extract_weight(weight_str):
#     weight_values = []
#     if isinstance(weight_str, str):
#         for w in weight_str.split('/'):
#             for sub_w in w.split('or'):
#                 sub_w = re.sub(r'\([^()]*\)', '', sub_w)  # 删除括号及其中的内容
#                 sub_w = sub_w.strip().split()[0]  # 只取重量值中的第一个数字部分
#                 sub_w = sub_w.replace('g', '')  # 去除"g"字符
#                 if sub_w.replace('.', '', 1).isdigit():  # 判断是否为数字，允许有一个小数点
#                     weight_values.append(float(sub_w))  # 提取所有重量值
#     return sum(weight_values) / len(weight_values) if weight_values else None

# df['Body_Weight_gram'] = df['Body_Weight'].apply(extract_weight)

# def extract_dimensions(dimensions_str):
#     if isinstance(dimensions_str, str):
#         if 'fold' in dimensions_str.lower():
#             return pd.Series([None, None, None])
#         else:
#             dimensions_parts = dimensions_str.split('(')[0].split('x')
#             dimensions_mm = [part.strip().split()[0] for part in dimensions_parts[:3]]  # 取前3个尺寸的数字部分
#             return pd.Series(dimensions_mm)
#     else:
#         return pd.Series([None, None, None])


# df[['Length', 'Width', 'Height']] = df['Body_Dimensions'].apply(extract_dimensions)

# def remove_dash(value):
#     if isinstance(value, str):
#         return value.replace('-', '')
#     else:
#         return value

# 将 remove_dash 函数应用到每个元素
# df = df.applymap(remove_dash)
# df['Length']=pd.to_numeric(df['Length'], errors='coerce')
# df['Width']=pd.to_numeric(df['Width'], errors='coerce')
# df['Height']=pd.to_numeric(df['Height'], errors='coerce')




# def parse_date(date_str):
#     if not isinstance(date_str, str):
#         return None
#     try:
#         date = pd.to_datetime(date_str)
#         return date.strftime("%Y-%m-%d")
#     except (ValueError, TypeError):
#         pass
#     # 定义日期格式的正则表达式
#     patterns = [
#         r'Released\s(\d{4}),\s(\w+)\s(\d{1,2})',        # 2: Released 2010, March (第二个日期)
#         r'(\d{4}),\s(Q\d)',                             # 4: 2017, Q1
#         r'(\d{4}),\s(\w+)',                             # 5: 2006, June
#     ]

#     for pattern in patterns:
#         match = re.search(pattern, date_str)
#         if match:
#             if len(match.groups()) == 3:
#                 year, month, day = match.groups()
#                 date = datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
#                 return date.strftime("%Y-%m-%d")
#             elif len(match.groups()) == 2:
#                 year, month_or_quarter = match.groups()
#                 if 'Q' in month_or_quarter:
#                     month_day = {
#                         'Q1': '01-01',
#                         'Q2': '04-01',
#                         'Q3': '07-01',
#                         'Q4': '10-01'
#                     }
#                     date = f"{year}-{month_day[month_or_quarter]}"
#                 else:
#                     date = datetime.strptime(f"01 {month_or_quarter} {year}", "%d %B %Y")
#                     date = date.strftime("%Y-%m-%d")
#                 return date

#     if "Not" in date_str:
#         return None

#     return None

# 读取Excel文件并解析日期

# df['parsed_date'] = df['Launch_Announced'].apply(parse_date)


# 去除2007年1月前发售的手机
# df = df[df['parsed_date'] >= '2007-01-01']



# 去除重量大于317g的手机，但保留空值
# df = df[(df['Body_Weight_gram'] <= 317) | (df['Body_Weight_gram'].isnull())]
#去除三星手表
# df = df[df['型号'].str.contains('Samsung Gear', case=False)==False]

#去除重量小于50g的手机
# df = df[(df['Body_Weight_gram'] >= 50)| (df['Body_Weight_gram'].isnull())]


# Extract numeric data from 'Display_Size', 'Display_Resolution', 'screentobody ratio' and 'ppi density'
# Extract numeric data from 'Display_Size', 'Display_Resolution', 'screentobody ratio' and 'ppi density'
# df['Size_Inches'] = df['Display_Size'].str.extract(r'(\d+\.\d+)').astype(float)
# resolution_data = df['Display_Resolution'].str.extract(r'(\d+) x (\d+)')
# df['Resolution_Width'] = pd.to_numeric(resolution_data[0], errors='coerce')
# df['Resolution_Height'] = pd.to_numeric(resolution_data[1], errors='coerce')

# # Extract screen-to-body ratio
# df['Screen_To_Body_Ratio'] = df['Display_Size'].str.extract(r'(\d+\.\d+)%').astype(float)

# # Extract ppi density
# ppi_data = df['Display_Resolution'].str.extract(r'(~\d+ ppi)')
# df['PPI_Density'] = ppi_data[0].str.extract(r'(\d+)').astype(float)


all_parts = set()  
for item in df['Network_Technology']:  
    parts = str(item).split('/')  
    for part in parts:  
        part = part.strip().upper()  # 去除每个part前后的空白字符  
        if part:  # 忽略空字符串  
            all_parts.add(part)  
print(all_parts)
# 为每个唯一的part创建新列，并初始化为0  
for part in all_parts:  
    df[part] = 0  
  
# 填充新列的值  
for index, item in df['Network_Technology'].items():  
    parts = str(item).split('/')  
    for part in parts:  
        part = part.strip()   
        df.at[index, part.upper()] = 1  # 使用.at[]来设置单个值  
  
df.to_excel(r"D:\personal\2024\py\eda\2150308_EDA_homework\data\phone_parameters_refined.xlsx", index=False)

df.info()

