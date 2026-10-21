import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from streamlit_drawable_canvas import st_canvas

# -------------------- بارگذاری مدل --------------------
@st.cache_resource
def load_model():
    return joblib.load("models/voting_model.joblib")

model = load_model()
digits = load_digits()

st.set_page_config(page_title="تشخیص رقم دست‌نویس", page_icon="✍️", layout="centered")
st.title("✍️ تشخیص رقم دست‌نویس")
st.markdown(
    "رقم را با **ماوس بکش**، یا **تصویر آپلود کن**، یا از **نمونه‌های آماده** استفاده کن."
)

# -------------------- پیش‌پردازش قوی --------------------
def to_digits_format(img_input):
    """
    تبدیل تصویر (PIL یا numpy) به بردار ۶۴تایی سازگار با sklearn digits
    (۸×۸، مقادیر تقریباً ۰ تا ۱۶، رقم روشن روی پس‌زمینهٔ تیره)
    """
    # --- به grayscale و numpy ---
    if isinstance(img_input, Image.Image):
        img = img_input.convert("L")
        arr = np.array(img).astype(np.float64)
    else:
        arr = np.array(img_input).astype(np.float64)
        if arr.ndim == 3:
            # RGBA یا RGB
            if arr.shape[2] == 4:
                # کانال آلفا را هم در نظر بگیر (قلم روی شفاف)
                alpha = arr[:, :, 3] / 255.0
                gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
                arr = gray * alpha  # پس‌زمینه شفاف ≈ ۰
            else:
                arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    if arr.max() == arr.min():
        return np.zeros((1, 64))

    # نرمال به ۰–۲۵۵
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0

    # اگر پس‌زمینه روشن و رقم تیره است → برعکس کن (می‌خواهیم رقم روشن باشد)
    # معیار: میانگین لبه‌ها
    h, w = arr.shape
    border = np.concatenate([
        arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]
    ])
    if border.mean() > 127:
        arr = 255.0 - arr

    # آستانه‌گذاری ملایم برای حذف نویز
    arr = np.where(arr > 30, arr, 0)

    # پیدا کردن جعبهٔ محدودکنندهٔ رقم
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1, 64))

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    cropped = arr[y1 : y2 + 1, x1 : x2 + 1]

    # مربع کردن با حفظ نسبت و کمی حاشیه
    ch, cw = cropped.shape
    side = max(ch, cw)
    pad = int(side * 0.2)  # ۲۰٪ حاشیه مثل دیتاست digits
    side = side + 2 * pad
    square = np.zeros((side, side), dtype=np.float64)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    square[y_off : y_off + ch, x_off : x_off + cw] = cropped

    # تغییر اندازه به ۸×۸ با فیلتر نرم
    pil = Image.fromarray(square.astype(np.uint8))
    pil = pil.resize((8, 8), Image.Resampling.LANCZOS)
    out = np.array(pil).astype(np.float64)

    # اسکیل به بازهٔ تقریبی ۰–۱۶ (مثل load_digits)
    if out.max() > 0:
        out = out / out.max() * 16.0

    return out.flatten().reshape(1, -1)


def show_result(pred, proba, arr, source=""):
    confidence = proba[pred] * 100
    st.success(f"### رقم تشخیص‌داده‌شده: **{pred}**")
    if source:
        st.caption(f"منبع: {source}")
    st.metric("درصد اطمینان", f"{confidence:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["#4CAF50" if i == pred else "#2196F3" for i in range(10)]
    bars = ax.bar(range(10), proba * 100, color=colors)
    ax.set_xticks(range(10))
    ax.set_xlabel("رقم")
    ax.set_ylabel("احتمال (%)")
    ax.set_title("توزیع احتمال هر رقم")
    ax.set_ylim(0, 100)
    for bar, p in zip(bars, proba):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{p * 100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    st.pyplot(fig)
    plt.close(fig)

    st.caption("تصویر بعد از پیش‌پردازش (۸×۸):")
    display = (arr.reshape(8, 8) / 16.0 * 255).astype(np.uint8)
    st.image(display, width=120)


# ==================== ۱. کشیدن با ماوس ====================
st.subheader("۱. کشیدن رقم با ماوس")
st.caption("با ماوس روی کادر سیاه رقم را بکش (قلم سفید). بعد دکمهٔ تشخیص را بزن.")

stroke_width = st.slider("ضخامت قلم", 8, 40, 18)

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
        # image_data شکل (H, W, 4) RGBA است
        arr = to_digits_format(canvas_result.image_data)
        if arr.sum() == 0:
            st.warning("چیزی کشیده نشده یا خیلی کمرنگ است. دوباره امتحان کن.")
        else:
            pred = model.predict(arr)[0]
            proba = model.predict_proba(arr)[0]
            show_result(pred, proba, arr, source="کشیدن با ماوس")
    else:
        st.warning("هنوز چیزی نکشیده‌ای.")


# ==================== ۲. آپلود تصویر ====================
st.subheader("۲. آپلود تصویر رقم")
uploaded_file = st.file_uploader(
    "تصویر رقم (png / jpg) — ترجیحاً رقم تیره روی پس‌زمینه روشن",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, width=200, caption="تصویر آپلودشده")

    if st.button("تشخیص رقم آپلودشده", use_container_width=True):
        arr = to_digits_format(img)
        if arr.sum() == 0:
            st.warning("رقم در تصویر پیدا نشد.")
        else:
            pred = model.predict(arr)[0]
            proba = model.predict_proba(arr)[0]
            show_result(pred, proba, arr, source="تصویر آپلودشده")


# ==================== ۳. نمونه‌های آماده ====================
st.subheader("۳. نمونه‌های آماده از دیتاست")
sample_idx = st.slider("شماره نمونه", 0, len(digits.images) - 1, 42)
col_a, col_b = st.columns([1, 2])
with col_a:
    sample_img = (digits.images[sample_idx] / 16.0 * 255).astype(np.uint8)
    st.image(sample_img, width=120, caption=f"رقم واقعی: {digits.target[sample_idx]}")
with col_b:
    if st.button("تشخیص این نمونه", use_container_width=True):
        arr = digits.data[sample_idx].reshape(1, -1)
        pred = model.predict(arr)[0]
        proba = model.predict_proba(arr)[0]
        show_result(pred, proba, arr, source=f"نمونه شماره {sample_idx}")