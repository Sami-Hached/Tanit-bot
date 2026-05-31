import requests
import streamlit as st

API_URL = "http://localhost:8000/query/"

st.set_page_config(page_title="المساعد الذكي", layout="centered")

st.markdown(
    """
    <style>
        body, .stApp { direction: rtl; }
        textarea, .stTextArea textarea { direction: rtl; text-align: right; }
        .response-text {
            direction: rtl;
            text-align: right;
            font-size: 17px;
            line-height: 2;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("المساعد الذكي")

query = st.text_area("اكتب سؤالك هنا", height=120, placeholder="مثال: ما هي أفضل الممارسات لأمن كلمات المرور؟")

if st.button("إرسال", use_container_width=True):
    if not query.strip():
        st.warning("يرجى كتابة سؤال قبل الإرسال.")
    else:
        with st.spinner("جارٍ البحث عن إجابة..."):
            try:
                resp = requests.get(API_URL, params={"q": query}, timeout=30)
                resp.raise_for_status()
                answer = resp.json().get("response", "")
                st.divider()
                st.markdown(
                    f'<div class="response-text">{answer}</div>',
                    unsafe_allow_html=True,
                )
            except requests.exceptions.ConnectionError:
                st.error("تعذّر الاتصال بالخادم. تأكد من تشغيل FastAPI على المنفذ 8000.")
            except Exception as e:
                st.error(f"حدث خطأ: {e}")
