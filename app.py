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

st.set_page_config(page_title="Customer Support AI Assistant", page_icon="🤖", layout="centered")

GOOGLE_DRIVE_FILE_ID = "1zZPTJADFvfVh6DwTnRhEJgwkT7a4yiDV"

@st.cache_resource
def download_and_extract_models():
    target_dir = "Chatbot_Models"
    if not os.path.exists(target_dir):
        with st.spinner("Loading system models..."):
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
    lang_model = joblib.load(os.path.join(base_path, "lang_model", "model.pkl"))
    sent_path = os.path.join(base_path, "sentiment_model")
    sent_tokenizer = AutoTokenizer.from_pretrained(sent_path)
    sent_model = AutoModelForSequenceClassification.from_pretrained(sent_path)
    intent_model = joblib.load(os.path.join(base_path, "intent_model", "model.pkl"))
    rag_path = os.path.join(base_path, "rag_store")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    faiss_index = faiss.read_index(os.path.join(rag_path, "support_kb_index.faiss"))
    with open(os.path.join(rag_path, "support_responses.json"), "r", encoding="utf-8") as f:
        responses = json.load(f)
    return lang_model, sent_tokenizer, sent_model, intent_model, encoder, faiss_index, responses

lang_model, sent_tokenizer, sent_model, intent_model, encoder, faiss_index, responses = load_all_models()

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
client = Groq(api_key=groq_api_key) if groq_api_key else None

st.title("🤖 Customer Support AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "diagnostics" in message:
            with st.expander("تفاصيل التحليل الخلفي"):
                st.write(f"- اللغة المكتشفة: `{message['diagnostics']['lang']}`")
                st.write(f"- تصنيف النية: `{message['diagnostics']['intent']}`")
                st.write(f"- الحالة النفسية: `{message['diagnostics']['sent']}`")

if user_query := st.chat_input("اكتب استفسارك أو شكواك هنا..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    if not client:
        st.error("مفتاح Groq API غير متوفر.")
    else:
        with st.spinner("جاري المعالجة..."):
            lang_pred = lang_model.predict([user_query])[0]
            
            inputs = sent_tokenizer(user_query, return_tensors="pt", truncation=True, padding=True)
            outputs = sent_model(**inputs)
            sent_id = outputs.logits.argmax(dim=-1).item()
            sent_str = "Frustrated / Angry" if sent_id == 0 else "Neutral"
            
            intent_pred = intent_model.predict([user_query])[0]
            
            query_vector = encoder.encode([user_query], convert_to_numpy=True)
            distances, indices = faiss_index.search(query_vector, 2)
            retrieved_chunks = [responses[idx][:350] for idx in indices[0]]
            context = "\n".join([f"- {chunk.strip()}" for chunk in retrieved_chunks])
            
            prompt = (
                "System Instructions:\n"
                "You are a professional customer support assistant.\n"
                "1. Answer the customer query using ONLY the provided Context.\n"
                "2. LANGUAGE RULE: You MUST reply in the exact same language as the customer's question (If the customer writes in Arabic, you must reply in Arabic. If English, reply in English).\n"
                "3. FALLBACK RULE: If the context does not contain the answer, respond in the customer's language stating that you don't have the information and will connect them with an agent.\n"
                f"4. Customer Sentiment: {sent_str}. If the customer is Frustrated or Angry, begin your response with a polite apology acknowledging their frustration.\n\n"
                f"[Context]\n{context}\n\n"
                f"[Customer Question]\n\"{user_query}\"\n"
                "Answer:"
            )

            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b", 
                temperature=0.0,
                max_tokens=250
            )
            bot_response = chat_completion.choices[0].message.content

            diagnostics = {
                "lang": lang_pred,
                "intent": intent_pred,
                "sent": sent_str
            }
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_response,
                "diagnostics": diagnostics
            })
            
            with st.chat_message("assistant"):
                st.write(bot_response)
                with st.expander("تفاصيل التحليل الخلفي"):
                    st.write(f"- اللغة المكتشفة: `{lang_pred}`")
                    st.write(f"- تصنيف النية: `{intent_pred}`")
                    st.write(f"- الحالة النفسية: `{sent_str}`")
