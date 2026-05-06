import streamlit as st
from transformers import pipeline

# تحميل المودل وتخزينه في الكاش لتسريع التطبيق
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

# واجهة التطبيق
st.title("تطبيق تحليل المشاعر باستخدام BERT 🤖")
st.write("اكتب أي جملة باللغة الإنجليزية والمودل هيحدد إذا كانت المشاعر إيجابية أم سلبية.")

# تحميل المودل
model = load_model()

# مكان لإدخال النص
user_input = st.text_area("أدخل النص هنا:", "I am very happy today!")

# زرار التحليل
if st.button("تحليل النص"):
    if user_input:
        with st.spinner("جاري التحليل..."):
            result = model(user_input)
            
            # عرض النتيجة
            label = result[0]['label']
            score = result[0]['score']
            
            st.success(f"النتيجة: {label}")
            st.info(f"نسبة التأكد: {score:.2f}")
    else:
        st.warning("الرجاء إدخال نص أولاً!")
