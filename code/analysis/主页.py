import streamlit as st
from PIL import Image

st.set_page_config(page_title="主页", page_icon="👋",layout="wide")
st.title('手机发展趋势和关键变化')
img=Image.open("data/cover.jpg")
st.image(img)