import streamlit as st
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

st.set_page_config(page_title="数据总览", page_icon="🌍",layout="wide")

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

@st.cache_data
def load_data2(file_path):
    df = pd.read_excel(file_path)
    return df

@st.cache_data
def load_data3(file_path):
    df = pd.read_excel(file_path, sheet_name='Data', skiprows=4)
    return df

file_path = 'data/vendor-ww-monthly-201003-202405.csv'

data = load_data(file_path)                                                                       # 月度份额

excel_file = "data/phone_parameters.xlsx"
df_phone_models = load_data2(excel_file)                                                         # 手机参数数据

df_phone_models = df_phone_models.dropna(subset='型号')

excel_file = "data/phone_parameters_refined.xlsx"
df_phone_models2 = load_data2(excel_file)        

file_path_shipment_ww = 'data/statistic_id271490_global-smartphone-shipments-by-vendor-2009-2024.xlsx'
data_shipment_ww = load_data3(file_path_shipment_ww)              # 季度出货量数据

st.markdown('''
## 样本介绍

在进行手机发展分析时我选择了非随机抽样方法中的判断抽样（judgment sampling）。判断抽样是一种基于研究者专业判断和知识来选择样本的非概率抽样方法。在我的分析中，我决定选取知名品牌2007年后（第一部苹果手机问世后）发布的手机进行研究。

### 选择判断抽样的理由

1. **代表性**：知名品牌通常引领行业创新，可以更好地反映出市场的主流趋势和技术的进步。

2. **数据可获得性**：知名品牌的手机数据更容易获得且更完整，有助于提高分析的质量和可靠性。

3. **时间限制**：通过判断抽样，可以在有限的时间内选择质量最高的样本进行深入分析，从而获取有用的见解。

通过这种方法，我能够在有限的资源和时间内对手机市场的主要发展趋势进行全面的探索和分析，为后续的研究提供坚实的基础。
''')
st.divider()
st.markdown('## 数据总览')
st.markdown('### 手机参数数据')
st.write("来源：https://www.gsmarena.com/")
st.write(df_phone_models.head())
st.markdown('### 各供应商月度份额')
st.write("来源：https://gs.statcounter.com/")
st.write(data.head())
st.markdown('### 各供应商季度出货量')
st.write("来源：https://www.statista.com/")
st.write(data_shipment_ww.head())

st.divider()
st.markdown('## 缺失值可视化')
st.markdown('### 手机参数数据')
fig1, ax1 = plt.subplots(figsize=(10, 6))  # 调整图的比例
msno.matrix(df_phone_models, ax=ax1, sparkline=False)
st.pyplot(fig1)

st.markdown('### 各供应商月度份额')
fig2, ax2 = plt.subplots(figsize=(10, 6))  # 调整图的比例
msno.matrix(data, ax=ax2, sparkline=False)
st.pyplot(fig2)

st.markdown('### 各供应商季度出货量')
fig3, ax3 = plt.subplots(figsize=(10, 6))  # 调整图的比例
msno.matrix(data_shipment_ww, ax=ax3, sparkline=False)
st.pyplot(fig3)

st.divider()
st.markdown('## 数据变换')
st.markdown('### 类型转换')
def extract_weight(weight_str):
    weight_values = []
    if isinstance(weight_str, str):
        for w in weight_str.split('/'):
            for sub_w in w.split('or'):
                sub_w = re.sub(r'\([^()]*\)', '', sub_w)  # 删除括号及其中的内容
                sub_w = sub_w.strip().split()[0]  # 只取重量值中的第一个数字部分
                sub_w = sub_w.replace('g', '')  # 去除"g"字符
                if sub_w.replace('.', '', 1).isdigit():  # 判断是否为数字，允许有一个小数点
                    weight_values.append(float(sub_w))  # 提取所有重量值
    return sum(weight_values) / len(weight_values) if weight_values else None

df_phone_models['Body_Weight_gram'] = df_phone_models['Body_Weight'].apply(extract_weight)
df_show=df_phone_models[['Body_Weight_gram','Body_Weight']]
st.write(df_show.head())

st.markdown('### 维度拆分')
def extract_dimensions(dimensions_str):
    if isinstance(dimensions_str, str):
        if 'fold' in dimensions_str.lower():
            return pd.Series([None, None, None])
        else:
            dimensions_parts = dimensions_str.split('(')[0].split('x')
            dimensions_mm = [part.strip().split()[0] for part in dimensions_parts[:3]]  # 取前3个尺寸的数字部分
            return pd.Series(dimensions_mm)
    else:
        return pd.Series([None, None, None])
df_phone_models[['Length', 'Width', 'Height']] = df_phone_models['Body_Dimensions'].apply(extract_dimensions)
df_show=df_phone_models[['Body_Dimensions','Length', 'Width', 'Height']]
st.write(df_show.head())

st.markdown('### 日期正则化')
def parse_date(date_str):
    if not isinstance(date_str, str):
        return None
    try:
        date = pd.to_datetime(date_str)
        return date.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        pass
    # 定义日期格式的正则表达式
    patterns = [
        r'Released\s(\d{4}),\s(\w+)\s(\d{1,2})',        # 2: Released 2010, March (第二个日期)
        r'(\d{4}),\s(Q\d)',                             # 4: 2017, Q1
        r'(\d{4}),\s(\w+)',                             # 5: 2006, June
    ]

    for pattern in patterns:
        match = re.search(pattern, date_str)
        if match:
            if len(match.groups()) == 3:
                year, month, day = match.groups()
                date = datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
                return date.strftime("%Y-%m-%d")
            elif len(match.groups()) == 2:
                year, month_or_quarter = match.groups()
                if 'Q' in month_or_quarter:
                    month_day = {
                        'Q1': '01-01',
                        'Q2': '04-01',
                        'Q3': '07-01',
                        'Q4': '10-01'
                    }
                    date = f"{year}-{month_day[month_or_quarter]}"
                else:
                    date = datetime.strptime(f"01 {month_or_quarter} {year}", "%d %B %Y")
                    date = date.strftime("%Y-%m-%d")
                return date

    if "Not" in date_str:
        return None

    return None


df_phone_models['parsed_date'] = df_phone_models['Launch_Announced'].apply(parse_date)
df_show=df_phone_models[['Launch_Announced','parsed_date']]
st.write(df_show.head())


st.divider()

st.markdown('## 异常值分析及处理')

# 画boxenplot
fig4, ax4 = plt.subplots(figsize=(2, 2))
sns.boxenplot(y=df_phone_models['Body_Weight_gram'], ax=ax4)

col1, col2, col3 = st.columns([1, 2, 1])

# 在中间的列中显示图表
with col2:
    # 画boxenplot
    fig4, ax4 = plt.subplots(figsize=(5, 5))  # 调整图的绝对大小为6x4英寸
    sns.boxenplot(y=df_phone_models['Body_Weight_gram'], ax=ax4)
   
    st.pyplot(fig4, bbox_inches='tight')  # 使用bbox_inches='tight'来修剪图表的多余边框
st.markdown('#### 查看重量分布时发现有重量大于800g及小于50g的样本，查询后发现是型号中不含Tab、Tablet、Watch等的智能平板、智能手表等，进行手动删除。')
st.divider()
st.markdown('## 样本质量')
# 选择数值列
df_numeric = df_phone_models2.select_dtypes(include=['float64', 'int64'])
df_non_binary = df_numeric.loc[:, df_numeric.nunique() > 2]
# 计算每列的方差
variances = df_numeric.var()

# 在 Streamlit 中显示每列的方差（水平显示）
st.write("各数值列方差：")
st.write(variances.to_frame().T)

# 每行显示四个箱线图
columns = df_non_binary.columns
num_columns = len(columns)
num_plots_per_row = 4

for i in range(0, num_columns, num_plots_per_row):
    cols = st.columns(num_plots_per_row)
    for j, col in enumerate(cols):
        if i + j < num_columns:
            column = columns[i + j]
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.boxplot(y=df_numeric[column], ax=ax)
            ax.set_title(f'Boxplot of {column}')
            col.pyplot(fig)

# st.divider()
# st.markdown('## 5、变量选择')
# st.markdown('### 筛选出缺失值较少的特征')
# # 删除非数值列，以便计算方差
# df_numeric = df_phone_models2.select_dtypes(include=['float64', 'int64'])

# # 计算每列的方差
# variances = df_numeric.var()

# # 在 Streamlit 中显示方差
# st.write("各列方差：")
# st.write(variances)
# # 计算相关性矩阵
# correlation_matrix = df_numeric.corr()

# # 绘制热图
# fig5, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
# st.pyplot(fig5)