import streamlit as st
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# -------------------- بارگذاری مدل و ماسک ویژگی --------------------
@st.cache_resource
def load_model_and_mask():
    model = joblib.load("models/voting_model.joblib")
    mask = joblib.load("models/feature_mask.joblib")
    return model, mask

model, feature_mask = load_model_and_mask()

st.set_page_config(
    page_title="تشخیص رقم دست‌نویس (MNIST + الگوریتم ژنتیک)",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ تشخیص رقم دست‌نویس")
st.markdown(
    """
    **نسخه ارتقایافته**  
    داده: MNIST ۲۸×۲۸ + انتخاب ویژگی و تنظیم هایپرپارامتر با الگوریتم ژنتیک
    """
)

# -------------------- پیش‌پردازش ۲۸×۲۸ + اعمال ماسک --------------------
def to_mnist_format(img_input, feature_mask):
    """
    تبدیل تصویر به فرمت MNIST (۲۸×۲۸) و اعمال ماسک ویژگی انتخاب‌شده توسط GA
    """
    if isinstance(img_input, Image.Image):
        img = img_input.convert("L")
        arr = np.array(img).astype(np.float64)
    else:
        arr = np.array(img_input).astype(np.float64)
        if arr.ndim == 3:
            if arr.shape[2] == 4:  # RGBA
                alpha = arr[:, :, 3] / 255.0
                gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
                arr = gray * alpha
            else:
                arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    if arr.max() == arr.min():
        return np.zeros((1, int(feature_mask.sum())))

    # نرمال‌سازی به ۰-۲۵۵
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0

    # اگر پس‌زمینه روشن و رقم تیره است → معکوس کن
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if border.mean() > 127:
        arr = 255.0 - arr

    # حذف نویز ملایم
    arr = np.where(arr > 20, arr, 0)

    # پیدا کردن محدوده رقم
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1, int(feature_mask.sum())))

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    cropped = arr[y1:y2+1, x1:x2+1]

    # مربع کردن + حاشیه
    ch, cw = cropped.shape
    side = max(ch, cw)
    pad = int(side * 0.2)
    side = side + 2 * pad
    square = np.zeros((side, side), dtype=np.float64)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    square[y_off:y_off+ch, x_off:x_off+cw] = cropped

    # تغییر اندازه به ۲۸×۲۸
    pil = Image.fromarray(square.astype(np.uint8))
    pil = pil.resize((28, 28), Image.Resampling.LANCZOS)
    out = np.array(pil).astype(np.float32).flatten()

    # نرمال به ۰-۱
    if out.max() > 0:
        out = out / 255.0

    # اعمال ماسک ویژگی
    out = out[feature_mask.astype(bool)]

    return out.reshape(1, -1)


def show_result(pred, proba, source=""):
    confidence = proba[pred] * 100
    st.success(f"### رقم تشخیص‌داده‌شده: **{pred}**")
    if source:
        st.caption(f"منبع: {source}")
    st.metric("درصد اطمینان", f"{confidence:.1f}%")

    # نمودار احتمال‌ها
    fig, ax = plt.subplots(figsize=(9, 3.6))
    colors = ["#4CAF50" if i == pred else "#2196F3" for i in range(10)]
    bars = ax.bar(range(10), proba * 100, color=colors)
    ax.set_xticks(range(10))
    ax.set_xlabel("رقم")
    ax.set_ylabel("احتمال (%)")
    ax.set_title("توزیع احتمال هر رقم")
    ax.set_ylim(0, 105)

    for bar, p in zip(bars, proba):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{p*100:.0f}%", ha="center", va="bottom", fontsize=9)

    st.pyplot(fig)
    plt.close(fig)


# ==================== ۱. کشیدن با ماوس ====================
st.subheader("۱. کشیدن رقم با ماوس")
st.caption("با ماوس روی کادر سیاه رقم را بکش (قلم سفید). سپس دکمه تشخیص را بزن.")

stroke_width = st.slider("ضخامت قلم", 12, 45, 24)

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="digit_canvas",
    update_streamlit=True,
    return_image_data=True
)

if st.button("تشخیص رقم کشیده‌شده", type="primary", use_container_width=True):
    if canvas_result.image_data is not None:
        arr = to_mnist_format(canvas_result.image_data, feature_mask)
        if arr.sum() == 0:
            st.warning("چیزی کشیده نشده یا خیلی کمرنگ است.")
        else:
            pred = model.predict(arr)[0]
            proba = model.predict_proba(arr)[0]
            show_result(pred, proba, source="کشیدن با ماوس")
    else:
        st.warning("هنوز چیزی نکشیده‌ای.")


# ==================== ۲. آپلود تصویر ====================
st.subheader("۲. آپلود تصویر رقم")
uploaded_file = st.file_uploader(
    "تصویر رقم را انتخاب کن (png یا jpg)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, width=220, caption="تصویر آپلودشده")

    if st.button("تشخیص رقم آپلودشده", use_container_width=True):
        arr = to_mnist_format(img, feature_mask)
        if arr.sum() == 0:
            st.warning("رقم در تصویر پیدا نشد.")
        else:
            pred = model.predict(arr)[0]
            proba = model.predict_proba(arr)[0]
            show_result(pred, proba, source="تصویر آپلودشده")


# ==================== اطلاعات مدل ====================
with st.expander("اطلاعات مدل و الگوریتم ژنتیک"):
    st.markdown(f"""
    - **داده آموزشی**: MNIST (۲۸×۲۸)
    - **تعداد ویژگی اصلی**: ۷۸۴
    - **تعداد ویژگی انتخاب‌شده توسط الگوریتم ژنتیک**: **{int(feature_mask.sum())}**
    - **مدل نهایی**: Voting Classifier  
      (Logistic Regression + KNN + SVM + Decision Tree)
    - هایپرپارامترها نیز با الگوریتم ژنتیک بهینه شده‌اند.
    """)
