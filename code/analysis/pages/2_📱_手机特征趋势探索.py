import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re

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

excel_file = "data/phone_parameters_refined.xlsx"
df = load_data2(excel_file)

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
st.markdown("## 2. 机型与屏幕保护特色")
df=df.sort_values(by='Year')

@st.cache_data
def generate_wordcloud(text):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        colormap='winter',
        contour_color='steelblue',
        contour_width=1
    ).generate(text)
    return wordcloud

years = df['Year'].unique()
selected_year = st.selectbox('Select Year', years)

if selected_year:
    text_data = ' '.join(df[df['Year'] == selected_year][['Body_Build']].fillna('').astype(str).apply(', '.join, axis=1))
    wordcloud = generate_wordcloud(text_data)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Year {selected_year}')    
    st.pyplot(plt)

    text_data2 = ' '.join(df[df['Year'] == selected_year][['Display_Protection']].fillna('').astype(str).apply(', '.join, axis=1))
    wordcloud = generate_wordcloud(text_data2)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Year {selected_year}')
    st.pyplot(plt)
    text_data3 = ' '.join(df[df['Year'] == selected_year][['Body_','Features_Sensors']].fillna('').astype(str).apply(', '.join, axis=1))
    wordcloud = generate_wordcloud(text_data3)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Year {selected_year}')
    st.pyplot(plt)

st.divider()


st.markdown("## 3. CPU和GPU型号趋势")
def extract_features(text):
    try:
        # 按空格和左括号分割
        parts = re.split(r'[ \(]', text)
        
        # 查找包含 "core" 的部分
        core_type = next((part for part in parts if 'core' in part.lower()), None)

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

# 对 df 的 Model 列应用提取特征函数
features = df['Platform_CPU'].apply(lambda x: extract_features(x))
df['Core Type'] = features.apply(lambda x: x[0] if x else None)
df['Max Frequency (GHz)'] = features.apply(lambda x: x[1] if x else None)
df = df[df['Max Frequency (GHz)'] <= 5]
yearly_brand_max_freq = df.groupby(['Year', 'Brand'])['Max Frequency (GHz)'].max().unstack()

# 时钟速度趋势图
fig8 = go.Figure()
for brand in yearly_brand_max_freq.columns:
    fig8.add_trace(go.Scatter(x=yearly_brand_max_freq.index, y=yearly_brand_max_freq[brand], mode='lines+markers', name=brand))

fig8.update_layout(
    title='Trend of Max Clock Speed by Year and Brand',
    xaxis_title='Year',
    yaxis_title='Max Frequency (GHz)'
)
st.plotly_chart(fig8)
# 按年份统计不同核心类型的比例
yearly_core_type = df.groupby(['Year', 'Core Type']).size().unstack().fillna(0)
yearly_percentages = yearly_core_type.div(yearly_core_type.sum(axis=1), axis=0) * 100
fig9 = go.Figure()
for core_type in yearly_percentages.columns:
    fig9.add_trace(go.Bar(x=yearly_percentages.index, y=yearly_percentages[core_type], name=core_type))

fig9.update_layout(
    barmode='stack',
    title='Percentage of Different Core Types by Year',
    xaxis_title='Year',
    yaxis_title='Percentage (%)'
)
st.plotly_chart(fig9)   

st.divider()
st.markdown("## 4. 网络技术趋势")

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
    title='Percentage of Mobile Network Technologies Support by Year (Relative to Total Phones)',
    xaxis_title='Year',
    yaxis_title='Percentage (%)'
)

# 计算每种技术的归一化到100%的支持百分比
yearly_normalized_percentages = yearly_data.div(yearly_data.sum(axis=1), axis=0) * 100

# 绘制堆叠柱状图 (每种技术归一化到100%的百分比)
fig4 = go.Figure()
for tech in network_columns:
    fig4.add_trace(go.Bar(x=yearly_normalized_percentages.index, y=yearly_normalized_percentages[tech], name=tech))

fig4.update_layout(
    barmode='stack',
    title='Percentage of Mobile Network Technologies Support by Year (Normalized to 100%)',
    xaxis_title='Year',
    yaxis_title='Percentage (%)'
)
st.plotly_chart(fig3)
st.plotly_chart(fig4)
st.markdown("总体趋势显示，从2G和3G技术逐渐过渡到4G LTE技术，并且最近几年开始向5G技术发展。GSM和HSPA等较旧的技术逐渐被淘汰，而LTE和5G等新技术的采用率迅速增加。")
st.divider()
st.markdown("## 各品牌重量趋势图")
# 按年份分组并计算不同型号的数量
weight_trend = df.groupby(['Year', 'Brand'])['Body_Weight_gram'].mean().reset_index()

# 创建带有拖动条的 Plotly 图表
fig = go.Figure()

for brand in weight_trend['Brand'].unique():
    brand_data = weight_trend[weight_trend['Brand'] == brand]
    fig.add_trace(go.Scatter(x=brand_data['Year'], y=brand_data['Body_Weight_gram'], mode='lines+markers', name=brand, line=dict(color=brand_colors[brand])))

fig.update_layout(
    title='Average Body Weight Over Years by Brand',
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
    width=1000,  # 设置图表宽度
    height=600,  # 设置图表高度
)
for trace in fig.data:
    trace.line.width = 2


# 在 Streamlit 中显示图表
st.plotly_chart(fig)
st.divider()


st.markdown("## 6. 屏幕尺寸与重量联合图")
# 不同品牌不同颜色
df_sorted = df.sort_values(by='Year')  
# 现在，使用排序后的 DataFrame 来创建图表  
fig2 = px.scatter(df_sorted, y='Body_Weight_gram', x='Size_Inches', color='Brand',  
                  title='Body Weight vs. Body Size by Brand and Year',  
                  labels={'Body_Weight_gram': 'Body Weight (grams)', 'Size_Inches': 'Body Size (inches)'},  # 注意修正了 labels 中的 'Body_Size' 到 'Size_Inches'  
                  animation_frame='Year',  
                  color_discrete_map=brand_colors) 
# 设置图表大小
fig2.update_layout(width=1000, height=800,xaxis=dict(range=[0, 8]),yaxis=dict(range=[0, 300]))
fig2.update_traces(marker_size=10)
st.plotly_chart(fig2)


st.divider()
st.markdown("## 7. 摄像头像素")
# 提取摄像头特性的函数
def extract_camera_features(text):
    try:
        resolution = re.search(r'(\d+\.?\d*) MP', text)
        aperture = re.search(r'f/(\d+\.?\d*)', text)
        pixel_size = re.search(r'(\d+\.?\d*)µm', text)
        ois = 'OIS' in text or 'ois' in text
        return {
            'Resolution (MP)': float(resolution.group(1)) if resolution else None,
            'Aperture': float(aperture.group(1)) if aperture else None,
            'Pixel Size (µm)': float(pixel_size.group(1)) if pixel_size else None,
            'OIS': ois
            
        }
    except Exception as e:
        return {
            'Resolution (MP)': None,
            'Aperture': None,
            'Pixel Size (µm)': None,
            'OIS': None
        }

# 提取视频特性的函数
def extract_video_features(text):
    try:
        if '4K' in text:
            resolution = '4K'
        elif '1080p' in text:
            resolution = '1080p'
        elif '720p' in text:
            resolution = '720p'
        elif '480p' in text:
            resolution = '480p'
        elif '360p' in text:
            resolution = '360p'
        else:
            resolution = None
        frame_rates = re.findall(r'@(\d+)fps', text)
        max_frame_rate = max([int(rate) for rate in frame_rates]) if frame_rates else None        
        
        return {
            'Resolution': resolution,
            'Max Frame Rate': max_frame_rate,            
        }
    except Exception as e:
        return {
            'Resolution': None,
            'Max Frame Rate': None,
            
        }

# 应用提取函数并创建新的列
camera_features = df['MainCamera_Triple'].apply(lambda x: extract_camera_features(x))
video_features = df['MainCamera_Video'].apply(lambda x: extract_video_features(x))

# 将提取的特性展开并合并到原始 DataFrame 中
camera_df = pd.DataFrame(camera_features.tolist())
video_df = pd.DataFrame(video_features.tolist())

df = pd.concat([df, camera_df.add_prefix('Camera '), video_df.add_prefix('Video ')], axis=1)

st.write(df.head())

def prepare_ternary_data(filtered_df):
    filtered_df = filtered_df[['Camera Resolution (MP)', 'Camera Aperture', 'Camera Pixel Size (µm)','Video Resolution', 'Video Max Frame Rate', 'Camera OIS']].dropna()
    
    filtered_df['Camera OIS'] = filtered_df['Camera OIS'].apply(lambda x: 1 if x else 0)
    
    filtered_df = filtered_df.rename(columns={
        'Camera Resolution (MP)': 'Resolution (MP)',
        'Camera Aperture': 'Aperture',
        'Camera Pixel Size (µm)': 'Pixel Size (µm)',   
        'Video Resolution': 'Resolution',
        'Video Max Frame Rate': 'Max Frame Rate',
        'Camera OIS': 'OIS'
    })
    
    return filtered_df

# 绘制 Ternary 图的函数
def plot_ternary_charts(data):
    camera_fig = px.scatter_ternary(data, a='Resolution (MP)', b='Aperture', c='Pixel Size (µm)', title='Camera Features Ternary Plot')
    video_fig = px.scatter_ternary(data, a='Resolution', b='Max Frame Rate', c='OIS', title='Video Features Ternary Plot')

    video_fig.update_layout({
    'ternary': {
        'sum': 1,
        'aaxis': {'title': 'Max Frequency (GHz)'},
        'baxis': {'title': 'Camera Resolution (Normalized)'},
        'caxis': {'title': 'Video Resolution'}
    },
    'title': 'Ternary Plot of Phone Specifications'
})
    
    st.plotly_chart(camera_fig, use_container_width=True)
    st.plotly_chart(video_fig, use_container_width=True)


# Streamlit 应用
st.title('Camera and Video Features Analysis')

# 选择年份
year_options = df['Year'].unique()
selected_year = st.selectbox('Select Year', year_options)

# 过滤数据
filtered_df = df[df['Year'] == selected_year]

# 准备数据
camera_data= prepare_ternary_data(filtered_df)

resolution_mapping = {'360p': 0.25, '720p': 0.5, '1080p': 0.75, '4K': 1.0}
df['Video Resolution'] = df['Video Resolution'].map(resolution_mapping)
st.write(camera_data)
# 绘制图表
plot_ternary_charts(camera_data)


st.divider()
st.markdown("## 8. 电池趋势图")
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
fig6 = px.bar(battery_type_normalized, title='Normalized Stacked Bar Chart of Battery Types by Year',
              labels={'value': 'Proportion', 'Year': 'Year', 'variable': 'Battery Type'},
              barmode='stack')
fig6.update_layout(xaxis_title='Year', yaxis_title='Proportion')
st.plotly_chart(fig6)

# 按年度和品牌计算电池容量的平均值
battery_capacity_trend = battery_info.groupby(['Year', 'Brand'])['Capacity_mAh'].mean().unstack()

# 绘制年度品牌电池容量趋势图
fig7 = go.Figure()
for brand in battery_capacity_trend.columns:
    if battery_capacity_trend[brand].sum() > 0:  # 过滤掉总和为零的品牌
        fig7.add_trace(go.Scatter(x=battery_capacity_trend.index, y=battery_capacity_trend[brand],
                                  mode='lines+markers', name=brand))
fig7.update_layout(title='Trend of Battery Capacity by Year and Brand',
                   xaxis_title='Year', yaxis_title='Average Capacity (mAh)',
                   legend_title='Brand')
st.plotly_chart(fig7)


st.divider()
st.markdown("## 9、蓝牙发展趋势")
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
    fig.update_layout(barmode='stack', title=title, xaxis_title='Year', yaxis_title='Proportion')
    st.plotly_chart(fig)

# 绘制图表
plot_stacked_bar_chart(technology_trends_wlan_filtered, 'WLAN Technology Trends')
plot_stacked_bar_chart(technology_trends_bluetooth_filtered, 'Bluetooth Technology Trends')
# plot_stacked_bar_chart(technology_trends_positioning_filtered, 'Positioning Technology Trends')


st.markdown("## 不同年份发布新机颜色比例")
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
    title='Normalized Percentage of Phone Colors by Year',
    xaxis_title='Year',
    yaxis_title='Percentage (%)',
    legend_title='Colors',
    width=1000,
    height=600
    
)

st.plotly_chart(fig5)