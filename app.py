import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import io

@st.cache_resource
def load_model():
    return joblib.load("models/voting_model.joblib")

model = load_model()
digits = load_digits()

st.set_page_config(page_title="تشخیص رقم دست‌نویس", page_icon="✍️", layout="centered")
st.title("✍️ تشخیص رقم دست‌نویس")
st.markdown("رقم را **به صورت لاتین** بکش و به عنوان تصویر آپلود کن، یا از نمونه‌های آماده استفاده کن.")

# ----------------- بخش نمونه‌های آماده -----------------
st.subheader("۱. انتخاب نمونه آماده از دیتاست")
sample_idx = st.slider("شماره نمونه", 0, len(digits.images)-1, 42)
col_a, col_b = st.columns([1, 2])
with col_a:
    st.image(digits.images[sample_idx], width=120, caption=f"رقم واقعی: {digits.target[sample_idx]}")
with col_b:
    if st.button("تشخیص این نمونه", use_container_width=True):
        arr = digits.data[sample_idx].reshape(1, -1)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0]
        show_result(pred, proba, arr, source=f"نمونه شماره {sample_idx}")

# ----------------- بخش آپلود تصویر -----------------
st.subheader("۲. آپلود تصویر رقم دست‌نویس خودت")
uploaded_file = st.file_uploader("تصویر رقم را اینجا بکش و آپلود کن (png یا jpg)", type=["png", "jpg", "jpeg"])

def preprocess(img):
    img = img.convert("L")
    img = ImageOps.invert(img)          # اگر سفید روی سیاه بود
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float64")
    arr = (arr / 255.0) * 16.0
    return arr.flatten().reshape(1, -1)

def show_result(pred, proba, arr, source=""):
    confidence = proba[pred] * 100
    st.success(f"### رقم تشخیص داده‌شده: **{pred}**")
    if source:
        st.caption(f"منبع: {source}")
    st.metric("درصد اطمینان", f"{confidence:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4CAF50" if i == pred else "#2196F3" for i in range(10)]
    bars = ax.bar(range(10), proba * 100, color=colors)
    ax.set_xticks(range(10))
    ax.set_xlabel("رقم")
    ax.set_ylabel("احتمال (%)")
    ax.set_title("توزیع احتمال هر رقم")
    ax.set_ylim(0, 100)
    for bar, p in zip(bars, proba):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{p*100:.1f}%", ha="center", va="bottom", fontsize=9)
    st.pyplot(fig)

    st.caption("تصویر ورودی بعد از پیش‌پردازش (۸×۸):")
    st.image(arr.reshape(8, 8), width=150, clamp=True)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, width=200, caption="تصویر آپلود شده")
    
    if st.button("تشخیص رقم آپلود شده", type="primary", use_container_width=True):
        arr = preprocess(img)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0]
        show_result(pred, proba, arr, source="تصویر آپلود‌شده")