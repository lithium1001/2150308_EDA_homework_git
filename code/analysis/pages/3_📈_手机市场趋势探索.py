import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Line
from streamlit_echarts import st_pyecharts


st.set_page_config(page_title="手机市场趋势探索", page_icon="📈",layout="wide")

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

file_path = r'D:\personal\2024\py\eda\2150308_EDA_homework\data\vendor-ww-monthly-201003-202405.csv'
data = load_data(file_path)                                                                       # 月度份额

excel_file = r"D:\personal\2024\py\eda\2150308_EDA_homework\data\phone_parameters_refined.xlsx"
df_phone_models = load_data2(excel_file)                                                         # 手机参数数据

file_path_shipment_ww = r'D:\personal\2024\py\eda\2150308_EDA_homework\data\statistic_id271490_global-smartphone-shipments-by-vendor-2009-2024.xlsx'
data_shipment_ww = load_data3(file_path_shipment_ww)              # 季度出货量数据
df1=data_shipment_ww.copy()

brand_colors = {
    'Apple': '#A9E5BB',  
    'Samsung': '#1428a0', 
    'Xiaomi': '#dd5144',  
    'Huawei': '#403242', 
    'Oppo': '#FCF6B1', 
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

st.markdown("## 知名供应商出货量趋势（百万）")

data_shipment_ww = data_shipment_ww.dropna(axis=1, how='all').dropna(axis=0, how='all')
data_shipment_ww = data_shipment_ww.set_index(data_shipment_ww.columns[0])

# 提取时间和销量数据
list_time = data_shipment_ww.index.tolist()
sale_dict = data_shipment_ww.to_dict(orient="list")


c = (
    Line(init_opts=opts.InitOpts(width="1500px"))  
    .add_xaxis(xaxis_data=list_time)
    .set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        datazoom_opts=opts.DataZoomOpts(),
        legend_opts=opts.LegendOpts(pos_left='90%',pos_top='30%'),  
    )
    
)

for name, sale_list in sale_dict.items():
    c.add_yaxis(
        series_name=name,
        stack="总量（百万）",
        y_axis=sale_list,
        areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
        label_opts=opts.LabelOpts(is_show=False),
        is_smooth=False,
        color=brand_colors.get(name, None)
    )


st_pyecharts(c, height=700)
st.divider()
st.markdown("## 知名供应商市场份额")
data=data.sort_values(by='Date',ascending=True)
data['Date'] = pd.to_datetime(data['Date'])
selected_date=data['Date'].max()
selected_date = st.select_slider(
    "Select date",
    value=data['Date'].max(),
    options=data['Date'].unique(),
    format_func=lambda x: x.strftime('%Y-%m')
)

# 根据选择的时间点过滤数据
filtered_data = data[data['Date']==selected_date]

# melt数据
melted_data = filtered_data.melt(id_vars=['Date'], var_name='Vendor', value_name='Share')

@st.experimental_fragment
def draw_mosaic_plot(data):
    fig = px.treemap(
        data, 
        title=selected_date.strftime('%Y-%m') + "的手机供应商市场份额",
        path=['Vendor'], 
        values='Share', 
        color='Vendor',
        labels={'Share': 'Percentage'},
        color_discrete_map=brand_colors
    )
    
    fig.update_traces(
        texttemplate='%{label}<br>%{value:.2f}%', 
        textinfo='label+text+value',
        textposition='middle center',
        textfont_size=35
        
    )
    fig.update_layout(width=1300, height=800,title=dict(font=dict(size=40)))
    return fig

config = dict({'displayModeBar': False})
if not melted_data.empty:
    fig = draw_mosaic_plot(melted_data)
    st.plotly_chart(fig,**{'config': config})
else:
    st.write("No data available for the selected date.")

st.divider()
st.markdown("## 出货量、市场份额与供应商该季度发布的新机型")
st.markdown("### 横轴为出货量（百万），纵轴为市场份额，气泡大小为该季度发布的新机型数量")
df1 = df1.iloc[3:]
df1.columns = df1.columns.str.replace('*', '').str.title()
def convert_quarter(date_str):
    parts = str(date_str).split(' ')
    if len(parts) == 2:
        quarter, year = parts[0], parts[1]
        return f"{quarter} \\'"+"20"+year[1:]
    else:
        return None  

df1['Quarter1'] = df1['Quarter'].apply(convert_quarter)
df2=data.copy()
df2.columns = df2.columns.str.replace('*', '').str.title()
df2['Quarter1'] = pd.to_datetime(df2['Date']).dt.to_period('Q').astype(str).replace(r'(\d{4})Q(\d)', r'Q\2 \'\1', regex=True)
df2_quarterly = df2.groupby('Quarter1').mean().reset_index()
result = pd.merge(df1, df2_quarterly, on='Quarter1', how='inner',suffixes=('_x', '_y'))

df_phone_models['parsed_date'] = pd.to_datetime(df_phone_models['parsed_date'], errors='coerce')
df_phone_models['Quarter'] = df_phone_models['parsed_date'].dt.to_period('Q').sort_values()

df_model_counts = df_phone_models.groupby(['Brand', 'Quarter']).size().reset_index(name='Models_Released')
df_model_counts['Brand'] = df_model_counts['Brand'].str.title()

def convert_quarter2(date_str):
    parts = str(date_str).split('Q')
    if len(parts) == 2:
        quarter, year = parts[1], parts[0]
        return f"Q{quarter} \\'{year}"
    else:
        return None 
df_model_counts['Quarter'] = df_model_counts['Quarter'].apply(convert_quarter2)

standardized_data = []
for index, row in result.iterrows():
    for brand in ['Apple', 'Samsung', 'Xiaomi','Oppo','Vivo','Nokia','Huawei','Sony','Htc','Lenovo']:  
        standardized_data.append({
            'Quarter': row['Quarter1'],
            'Brand': brand,
            'Shipment': row[f'{brand}_x'],
            'Market_Share': row[f'{brand}_y'],
        })

standardized_df = pd.DataFrame(standardized_data)
standardized_df=pd.merge(standardized_df,df_model_counts,on=['Brand','Quarter'],how='left')
standardized_df=standardized_df.fillna(0)
fig3 = px.scatter(
    standardized_df,
    x='Shipment',
    y='Market_Share',
    color='Brand',
    size='Models_Released',
    size_max=100,
    hover_name='Brand',
    animation_frame='Quarter',
    labels={'Market_Share_x': '出货量', 'Market_Share_y': '平均市场份额'},
    color_discrete_map=brand_colors
)

fig3.update_layout(width=1200, height=700)
fig3.update_layout(
    xaxis=dict(
        range=[0,100],
        showgrid=True
    ),
    yaxis=dict(
        range=[0,100],
        showgrid=True
    )

)

st.plotly_chart(fig3)