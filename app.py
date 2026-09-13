import os
import json
import zipfile
import joblib
import faiss
import gdown
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(page_title="Customer Support AI Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Customer Support AI Assistant")
st.write("مرحباً بك! نظام خدمة العملاء الذكي المعتمد على 4 نماذج ذكاء اصطناعي.")


GOOGLE_DRIVE_FILE_ID = "1zZPTJADFvfVh6DwTnRhEJgwkT7a4yiDV"

@st.cache_resource
def download_and_extract_models():
    target_dir = "Chatbot_Models"
    if not os.path.exists(target_dir):
        with st.spinner("جاري تحميل النماذج من Google Drive (تحدث مرة واحدة فقط)..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            zip_path = "models.zip"
            gdown.download(url, zip_path, quiet=False)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            if os.path.exists(zip_path):
                os.remove(zip_path)
    return True

@st.cache_resource
def load_all_models():
    download_and_extract_models()
    
    base_path = "Chatbot_Models"
    
    # Module 1: Language Detection
    lang_model = joblib.load(os.path.join(base_path, "lang_model", "model.pkl"))
    
    # Module 2: Sentiment Model
    sent_path = os.path.join(base_path, "sentiment_model")
    sent_tokenizer = AutoTokenizer.from_pretrained(sent_path)
    sent_model = AutoModelForSequenceClassification.from_pretrained(sent_path)
    
    # Module 3: Intent Classification
    intent_model = joblib.load(os.path.join(base_path, "intent_model", "model.pkl"))
    
    # Module 4: RAG Store
    rag_path = os.path.join(base_path, "rag_store")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index = faiss.read_index(os.path.join(rag_path, "support_kb_index.faiss"))
    with open(os.path.join(rag_path, "support_responses.json"), "r", encoding="utf-8") as f:
        responses = json.load(f)
        
    return lang_model, sent_tokenizer, sent_model, intent_model, encoder, faiss_index, responses

with st.spinner("جاري تهيئة النماذج والبيئة التشغيلية..."):
    lang_model, sent_tokenizer, sent_model, intent_model, encoder, faiss_index, responses = load_all_models()

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "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"))

if not groq_api_key:
    st.warning("⚠️ يرجى إضافة مفتاح GROQ_API_KEY في إعدادات Secrets.")

client = Groq(api_key=groq_api_key) if groq_api_key else None

user_query = st.text_area("اكتب استفسارك أو شكواك هنا:", height=100)

if st.button("إرسال الشكوى / الاستفسار"):
    if not user_query.strip():
        st.warning("من فضلك ادخل نصاً للبحث.")
    elif not client:
        st.error("مفتاح Groq API غير متوفر.")
    else:
        with st.spinner("جاري معالجة طلبك عبر الموديولات الأربعة..."):
            # 1. Language Detection
            lang_pred = lang_model.predict([user_query])[0]
            
            # 2. Sentiment Classification
            inputs = sent_tokenizer(user_query, return_tensors="pt", truncation=True, padding=True)
            outputs = sent_model(**inputs)
            sent_id = outputs.logits.argmax(dim=-1).item()
            sent_str = "frustrated" if sent_id == 0 else "neutral"
            
            # 3. Intent Classification
            intent_pred = intent_model.predict([user_query])[0]
            
            # 4. RAG Retrieval
            query_vector = encoder.encode([user_query], convert_to_numpy=True)
            distances, indices = faiss_index.search(query_vector, 2)
            retrieved_chunks = [responses[idx][:350] for idx in indices[0]]
            context = "\n".join([f"- {chunk.strip()}" for chunk in retrieved_chunks])
            
            # 5. Groq LLM Generation
            prompt = f"""[System Instructions]
You are a strict customer support assistant. 
Your ONLY job is to answer based EXCLUSIVELY on the provided Context.
Rules:
1. If context does not contain the answer, reply EXACTLY with: "I apologize, but I don't have that information. Let me connect you with an agent." DO NOT add anything else.
2. NEVER use general knowledge.
3. Customer Sentiment: {sent_str}. If frustrated, start with a polite apology.

[Context]
{context}

[Customer Question]
"{user_query}"
Answer:"""

            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
                temperature=0.0,
                max_tokens=250 
            )
            bot_response = chat_completion.choices[0].message.content
            
            st.success("تم التوليد بنجاح!")
            st.markdown("### 💬 إجابة المساعد الذكي:")
            st.write(bot_response)
            
            with st.expander("🔍 تفاصيل التحليل الخلفي (Pipeline Diagnostics)"):
                st.write(f"- **اللغة المكتشفة:** `{lang_pred}`")
                st.write(f"- **تصنيف النية (Intent):** `{intent_pred}`")
                st.write(f"- **الحالة النفسية (Sentiment):** `{sent_str}`")