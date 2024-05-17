import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
import numpy as np


st.set_page_config(page_title="手机特征趋势探索", page_icon="📱",layout="wide")

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

@st.experimental_fragment
def generate_wordcloud(text,colormap='winter'):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        colormap=colormap,
        contour_color='steelblue',
        contour_width=1
    ).generate(text)
    return wordcloud

# @st.cache_data
# def generate_wordcloud(text):
#     stopwords = set(STOPWORDS)
#     wordcloud = WordCloud(
#         width=800,
#         height=400,
#         background_color='white',
#         stopwords=stopwords,
#         colormap='winter',
#         contour_color='steelblue',
#         contour_width=1
#     ).generate(text)
#     return wordcloud

def extract_features(text):
    try:
        parts = re.split(r'[ \(]', text)
        
        # 查找包含 "core" 的部分
        core_type = next((part for part in parts if 'core' in part.lower()), None).title()

        # 查找频率并处理 GHz 和 MHz 单位
        freq_matches = re.findall(r"([\d\.]+) (GHz|MHz)", text, re.IGNORECASE)
        if freq_matches:
            freqs_in_ghz = [float(freq) / 1000 if unit.lower() == "mhz" else float(freq) for freq, unit in freq_matches]
            max_freq = max(freqs_in_ghz)
        else:
            max_freq = None

        return core_type, max_freq
    except Exception as e:
        return None, None

excel_file = "data/phone_parameters_refined.xlsx"
df = load_data2(excel_file)

excel_file = "data/cam.xlsx"
cam = load_data2(excel_file)

df['parsed_date'] = pd.to_datetime(df['parsed_date'], errors='coerce')

df['Year'] = df['parsed_date'].dt.year

brand_colors = {
    'Apple': '#A9E5BB',  
    'Samsung': '#1428a0', 
    'Xiaomi': '#dd5144',  
    'Huawei': '#403242', 
    'OPPO': '#FCF6B1', 
    'Nokia': '#5b68ac',  
    'Vivo': '#927396',  
    'HTC': '#757575',  
    'Lenovo': '#e57373', 
    'Google': '#4285f4',  
    'Sony': '#000000',  
    'Honor': '#546e7a',  
    'Realme': '#ffb74d', 
    'LG':'#c32f27',
    'Motorola':'#f0c929',
    'RIM':'#708090',
}

color_map = {
    'black': '#595959',
    'white': '#FBFCF7',
    'blue': '#62C5DA',
    'gold': '#FEEB75',
    'silver': '#C6C6D0',
    'red': '#FD6F5F',
    'gray': '#6A6880',
    'green': '#B2D3C2',
    'pink': '#F69ABF',
    'purple': '#DDA0DD',
    'rose gold': '#E6B8B7',
    'yellow': '#FFFFE0',
    'orange': '#FFDAB9',
    'pearl white': '#FBFCF7',
    'aurora': '#E6E6FA',
    'ocean blue': '#B0E0E6',
    'emerald green': '#98FB98',
    'violet': '#EE82EE',
    'sapphire blue': '#48AAAD',
    'mint': '#98EDC3'
}

st.markdown("## 代表机型")
img=Image.open("data/phone_development.jpg")
fig11=px.imshow(img)
fig11.update_layout(width=1000, height=800)
st.plotly_chart(fig11)
st.divider()
st.markdown("## 机身特色")
df=df.sort_values(by='Year')

years = df['Year'].unique()
selected_year=years.max()
selected_year = st.select_slider(
    label='Select Year',
    value=years.max(),
    options=years,    
)
# selected_year = st.selectbox('Select Year', years,key='wordcloud')

if selected_year:
    text_data1 = ' '.join(df[df['Year'] == selected_year][['Body_Build']].fillna('').astype(str).apply(', '.join, axis=1))
    text_data2 = ' '.join(df[df['Year'] == selected_year][['Display_Protection']].fillna('').astype(str).apply(', '.join, axis=1))
    # text_data3 = ' '.join(df[df['Year'] == selected_year][['Body_', 'Features_Sensors']].fillna('').astype(str).apply(', '.join, axis=1))

    wordcloud1 = generate_wordcloud(text_data1)
    wordcloud2 = generate_wordcloud(text_data2, colormap='BrBG')
    # wordcloud3 = generate_wordcloud(text_data3)
    
    # col1, col2, col3 = st.columns(3)
    col1, col2 = st.columns(2)

    with col1:
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud1, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{selected_year} Body Build')    
        st.pyplot(plt)
    
    with col2:
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud2, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{selected_year} Display Protection')
        st.pyplot(plt)
    
    # with col3:
    #     plt.figure(figsize=(10, 5))
    #     plt.imshow(wordcloud3, interpolation='bilinear')
    #     plt.axis('off')
    #     plt.title(f'Word Cloud for Body & Sensors {selected_year}')
    #     st.pyplot(plt)


st.divider()
st.markdown("## 新机型颜色比例趋势")
# 加载颜色计数数据
color_count_file_path = 'data/color_count.csv'
color_count_df = pd.read_csv(color_count_file_path)

# 筛选出热门颜色列表
top_colors = color_count_df.drop(columns=['Year', 'Unnamed: 1']).sum().sort_values(ascending=False).head(20)
top_color_list = top_colors.index.tolist()

# 将颜色名称标准化函数
def standardize_color(color_name):
    color_name = color_name.strip().lower()
    for color in top_color_list:
        if color in color_name:
            return color
    return 'unknown'

expanded_rows = []
for _, row in df[['Year', 'Misc_Colors']].dropna().iterrows():
    year = row['Year']
    colors = row['Misc_Colors'].split(',')
    for color in colors:
        cleaned_color = standardize_color(color)
        expanded_rows.append({'Year': year, 'Color': cleaned_color})

expanded_df = pd.DataFrame(expanded_rows)

# 统计每年每种颜色的出现次数
color_counts = expanded_df.groupby(['Year', 'Color']).size().unstack(fill_value=0)

# 计算每年的颜色比例
color_percentages = color_counts.div(color_counts.sum(axis=1), axis=0) * 100
missing_colors = set(top_color_list) - set(color_percentages.columns)
for color in missing_colors:
    color_percentages[color] = 0

# 绘制归一化的堆叠柱状图
fig5 = go.Figure()
for color in top_color_list:
    fig5.add_trace(go.Bar(
        x=color_percentages.index,
        y=color_percentages[color],
        name=color,
        marker_color=color_map.get(color, '#D3D3D3')  # 使用默认颜色
    ))

fig5.update_layout(
    barmode='stack',
    xaxis_title='Year',
    yaxis_title='Percentage (%)',
    legend_title='Colors',
    width=1100,
    height=700
    
)

st.plotly_chart(fig5)
st.divider()

st.markdown("## CPU发展趋势")


# 对 df 的 Model 列应用提取特征函数
features = df['Platform_CPU'].apply(lambda x: extract_features(x))
df['Core Type'] = features.apply(lambda x: x[0] if x else None)
df['Max Frequency (GHz)'] = features.apply(lambda x: x[1] if x else None)
df = df[df['Max Frequency (GHz)'] <= 5]
yearly_brand_max_freq = df.groupby(['Year', 'Brand'])['Max Frequency (GHz)'].max().unstack()

fig8 = go.Figure()
for brand in yearly_brand_max_freq.columns:
    fig8.add_trace(go.Scatter(x=yearly_brand_max_freq.index, y=yearly_brand_max_freq[brand], mode='lines+markers', name=brand))

fig8.update_layout(
    title='各品牌每年推出机型的最高时钟速度',
    xaxis_title='Year',
    yaxis_title='Max Frequency (GHz)',
    width=1100, height=700
)
st.plotly_chart(fig8)
yearly_core_type = df.groupby(['Year', 'Core Type']).size().unstack().fillna(0)
yearly_percentages = yearly_core_type.div(yearly_core_type.sum(axis=1), axis=0) * 100
fig9 = go.Figure()
for core_type in yearly_percentages.columns:
    fig9.add_trace(go.Bar(x=yearly_percentages.index, y=yearly_percentages[core_type], name=core_type))

fig9.update_layout(
    barmode='stack',
    title='每年新机型的CPU核心类型比例',
    xaxis_title='Year',
    yaxis_title='Percentage (%)',
    width=1100, height=700
)
st.plotly_chart(fig9)   

st.divider()
st.markdown("## 网络技术趋势")
st.markdown("### 移动网络技术")
# 网络技术列
network_columns = ['LTE', 'EVDO', 'HSPA', 'NO CELLULAR CONNECTIVITY', '5G', 'GSM', 'CDMA2000', 'CDMA', 'UMTS']

st.markdown('''NO CELLULAR CONNECTIVITY：表示设备不支持蜂窝网络连接
            
GSM (Global System for Mobile Communications)：第二代移动通信技术，广泛用于全球的移动通信标准

UMTS (Universal Mobile Telecommunications System)：一种3G移动通信标准，基于GSM技术

CDMA2000：3G移动通信标准，主要在北美和部分亚洲国家使用

EVDO (Evolution-Data Optimized)：一种用于CDMA网络的3G标准，主要用于高速数据传输

HSPA (High Speed Packet Access)：一种增强的3G技术，提供更快的数据传输速度

LTE (Long Term Evolution)：4G无线通信标准，提供高速数据传输

CDMA (Code Division Multiple Access)：一种无线通信技术，允许多个用户共享相同的频率

5G：第五代移动通信技术，提供更快的数据传输速度和更低的延迟''')

# 按年份计算每种技术的支持机型数量
yearly_data = df.groupby('Year')[network_columns].sum()

# 计算每年的手机总数
yearly_counts = df.groupby('Year').size()

# 计算每种技术的支持百分比，使用每年的手机总数作为分母
yearly_percentages = yearly_data.div(yearly_counts, axis=0) * 100

# 绘制堆叠柱状图 (每种技术相对于手机总数的百分比)
fig3 = go.Figure()
for tech in network_columns:
    fig3.add_trace(go.Bar(x=yearly_percentages.index, y=yearly_percentages[tech], name=tech))

fig3.update_layout(
    barmode='stack',
    title='每年新机型支持的移动网络技术比例',
    xaxis_title='Year',
    yaxis_title='Percentage (%)',
    width=1100, height=700
)

# 计算每种技术的归一化到100%的支持百分比
yearly_normalized_percentages = yearly_data.div(yearly_data.sum(axis=1), axis=0) * 100

# 绘制堆叠柱状图 (每种技术归一化到100%的百分比)
fig4 = go.Figure()
for tech in network_columns:
    fig4.add_trace(go.Bar(x=yearly_normalized_percentages.index, y=yearly_normalized_percentages[tech], name=tech))

fig4.update_layout(
    barmode='stack',
    title='每年新机型支持的移动网络技术比例 (归一化)',
    xaxis_title='Year',
    yaxis_title='Percentage (%)',
    width=1100, height=700
)
st.plotly_chart(fig3)
st.plotly_chart(fig4)
st.markdown("总体趋势显示，从2G和3G技术逐渐过渡到4G LTE技术，并且最近几年开始向5G技术发展。GSM和HSPA等较旧的技术逐渐被淘汰，而LTE和5G等新技术的采用率迅速增加。")

relevant_columns = ['parsed_date', 'Sound_35mmjack', 'Comms_WLAN', 'Comms_Bluetooth', 'Comms_Positioning', 'Comms_NFC']
df_relevant = df[relevant_columns]

relevant_columns = ['parsed_date', 'Comms_WLAN', 'Comms_Bluetooth', 'Comms_Positioning']
df_relevant = df[relevant_columns]

# 确保 'parsed_date' 列是 datetime 类型
df_relevant['parsed_date'] = pd.to_datetime(df_relevant['parsed_date'])

# 按年分组
df_relevant['Year'] = df_relevant['parsed_date'].dt.year

# 拆分技术字段
def split_and_expand(df, column, sep):
    return df.drop(column, axis=1).join(df[column].str.split(sep, expand=True).stack().reset_index(level=1, drop=True).rename(column))

df_wlan = split_and_expand(df_relevant, 'Comms_WLAN', ', ')
df_bluetooth = split_and_expand(df_relevant, 'Comms_Bluetooth', ', ')
# df_positioning = split_and_expand(df_relevant, 'Comms_Positioning', '; ')
# df_positioning = split_and_expand(df_positioning, 'Comms_Positioning', ', ')

# 按年分组并计算每种技术的比例
def resample_and_normalize(df, column):
    count_by_year = df.groupby(['Year', column]).size().unstack().fillna(0)
    normalized_by_year = count_by_year.div(count_by_year.sum(axis=1), axis=0)
    return normalized_by_year

technology_trends_wlan = resample_and_normalize(df_wlan, 'Comms_WLAN')
technology_trends_bluetooth = resample_and_normalize(df_bluetooth, 'Comms_Bluetooth')
# technology_trends_positioning = resample_and_normalize(df_positioning, 'Comms_Positioning')

# 过滤比例太小的技术
def filter_small_proportions(trends, threshold=0.05):
    filtered_trends = trends.loc[:, (trends > threshold).any()]
    return filtered_trends

technology_trends_wlan_filtered = filter_small_proportions(technology_trends_wlan)
technology_trends_bluetooth_filtered = filter_small_proportions(technology_trends_bluetooth)
# technology_trends_positioning_filtered = filter_small_proportions(technology_trends_positioning)

# 绘制堆叠柱状图
def plot_stacked_bar_chart(trends, title):
    fig = go.Figure()
    for column in trends.columns:
        fig.add_trace(go.Bar(x=trends.index, y=trends[column], name=column))
    fig.update_layout(barmode='stack', title=title, xaxis_title='Year', yaxis_title='Proportion',width=1100, height=700)
    st.plotly_chart(fig)

st.markdown("### WLAN技术")
plot_stacked_bar_chart(technology_trends_wlan_filtered, ' ')

st.divider()
st.markdown("## 电池发展趋势")
st.markdown("### 电池类型")
battery_data = df['Battery_Type'].dropna()
split_data = battery_data.str.split(',', n=1, expand=True)[0].str.extract(r'(\D+)\s(\d+)\s(\D+)')
split_data.columns = ['Battery_Type', 'Capacity_mAh', 'Removability']
split_data['Capacity_mAh'] = pd.to_numeric(split_data['Capacity_mAh'], errors='coerce')
# 合并处理后的电池数据
battery_info = pd.concat([df[['Year', 'Brand']], split_data], axis=1)
battery_info['Capacity_mAh'] = pd.to_numeric(battery_info['Capacity_mAh'], errors='coerce')

# 过滤掉没有年份或品牌的数据
battery_info = battery_info.dropna(subset=['Year', 'Brand'])

# 按年度统计不同类型电池的数量并进行归一化处理
battery_type_count = battery_info.groupby(['Year', 'Battery_Type']).size().unstack().fillna(0)
battery_type_normalized = battery_type_count.div(battery_type_count.sum(axis=1), axis=0)

# 绘制归一化堆叠柱状图
fig6 = px.bar(battery_type_normalized, 
              labels={'value': 'Proportion', 'Year': 'Year', 'variable': 'Battery Type'},
              barmode='stack')
fig6.update_layout(xaxis_title='Year', yaxis_title='Proportion',width=1100, height=700)
st.plotly_chart(fig6)

# 按年度和品牌计算电池容量的平均值
battery_capacity_trend = battery_info.groupby(['Year', 'Brand'])['Capacity_mAh'].mean().unstack()

st.markdown("### 电池容量")
fig7 = go.Figure()
for brand in battery_capacity_trend.columns:
    if battery_capacity_trend[brand].sum() > 0:  # 过滤掉总和为零的品牌
        fig7.add_trace(go.Scatter(x=battery_capacity_trend.index, y=battery_capacity_trend[brand],
                                  mode='lines+markers', name=brand))
fig7.update_layout(
                   xaxis_title='Year', yaxis_title='Average Capacity (mAh)',
                   width=1100, height=700,
                   legend_title='Brand')
st.plotly_chart(fig7)


st.divider()
st.markdown("## 蓝牙技术发展趋势")


# 绘制图表
plot_stacked_bar_chart(technology_trends_bluetooth_filtered, ' ')
# plot_stacked_bar_chart(technology_trends_positioning_filtered, 'Positioning Technology Trends')

st.divider()
def convert_resolution_to_numeric(resolution):
    if resolution == '4K':
        return 400
    elif resolution == '1080p':
        return 108
    elif resolution == '720p':
        return 72
    elif resolution == '480p':
        return 48
    elif resolution == '360p':
        return 36
    else:
        return None

cam['Resolution'] = cam['Resolution'].apply(convert_resolution_to_numeric)

# 定义雷达图的参数列
parameters = ['Resolution (MP)', 'Aperture', 'Pixel Size (µm)', 'Sensor Size (inches)', 'Resolution']

# Streamlit应用
st.markdown('## 手机摄像头参数')

selected_year = st.slider('选择年份', int(cam['parsed_date'].dt.year.min()), int(cam['parsed_date'].dt.year.max()), int(cam['parsed_date'].dt.year.max()), key='radar')

cam['Year'] = cam['parsed_date'].dt.year

# 过滤选定年份的数据
df_filtered = cam[cam['Year'] == selected_year]

# 计算每个品牌的平均值，忽略NaN值
df_avg = df_filtered.groupby('Brand')[parameters].mean().reset_index()


# 数据归一化
def normalize(df, parameters):
    df_normalized = df.copy()
    for parameter in parameters:
        min_val = df[parameter].min()
        max_val = df[parameter].max()
        df_normalized[parameter] = (df[parameter] - min_val) / (max_val - min_val)
    return df_normalized

df_normalized = normalize(df_avg, parameters)

@st.experimental_fragment
def radar_chart(df, parameters):
    fig = go.Figure()

    for i, row in df.iterrows():
        values = row[parameters].tolist()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=parameters + [parameters[0]],
            fill='toself',
            name=row['Brand']
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        width=1100,
        height=700
    )

    st.plotly_chart(fig)


# 显示雷达图
radar_chart(df_normalized, parameters)

selected_brand = st.selectbox('选择品牌', cam['Brand'].unique())

# 过滤选定品牌的数据
df_filtered = cam[cam['Brand'] == selected_brand]

df_avg = df_filtered.groupby('Year')[parameters].mean().reset_index()

# 绘制多重柱状图
@st.experimental_fragment
def bar_chart(df, parameters):
    fig = px.bar(
        df, 
        x='Year', 
        y=parameters, 
        barmode='group', 
        title=f'{selected_brand} 品牌摄像头参数',
        labels={'value': '参数值', 'variable': '参数'}
    )
    fig.update_layout(width=1100, height=700) 

    st.plotly_chart(fig)

# 显示多重柱状图
bar_chart(df_avg, parameters)
st.divider()
st.markdown("## 各品牌重量趋势图")
# 按年份分组并计算不同型号的数量
weight_trend = df.groupby(['Year', 'Brand'])['Body_Weight_gram'].mean().reset_index()

fig = go.Figure()

for brand in weight_trend['Brand'].unique():
    brand_data = weight_trend[weight_trend['Brand'] == brand]
    fig.add_trace(go.Scatter(x=brand_data['Year'], y=brand_data['Body_Weight_gram'], mode='lines+markers', name=brand, line=dict(color=brand_colors[brand])))

fig.update_layout(
    title='各品牌机型平均重量趋势',
    xaxis_title='Year',
    yaxis_title='Average Body Weight (grams)',
    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type='linear'
    ),
    
    font=dict(
        family="Arial, sans-serif",  # 使用清晰的字体
        size=14
    ),

    xaxis_tickformat='%Y-%m',  # 设置 X 轴刻度格式
    legend=dict(
        title='Brand',  # 图例标题
        font=dict(
            family="Arial, sans-serif",  # 图例字体
            size=12
        )
    ),
    width=1100,  # 设置图表宽度
    height=700,  # 设置图表高度
)
for trace in fig.data:
    trace.line.width = 2


# 在 Streamlit 中显示图表
st.plotly_chart(fig)
st.divider()


st.markdown("## 屏幕尺寸与重量发展趋势")
# 不同品牌不同颜色
df_sorted = df.sort_values(by='Year')  
# 现在，使用排序后的 DataFrame 来创建图表  
fig2 = px.scatter(df_sorted, y='Body_Weight_gram', x='Size_Inches', color='Brand',  
                  labels={'Body_Weight_gram': 'Body Weight (grams)', 'Size_Inches': 'Body Size (inches)'},  # 注意修正了 labels 中的 'Body_Size' 到 'Size_Inches'  
                  animation_frame='Year',  
                  color_discrete_map=brand_colors) 
# 设置图表大小
fig2.update_layout(width=1100, height=700,xaxis=dict(range=[0, 8]),yaxis=dict(range=[0, 300]))
fig2.update_traces(marker_size=10)
st.plotly_chart(fig2)


