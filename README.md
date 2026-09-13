# AI-Powered Customer Support Assistant

An intelligent customer support system powered by Natural Language Processing and Artificial Intelligence, featuring a 4-module pipeline for request handling, sentiment analysis, intent classification, and accurate response generation using Retrieval-Augmented Generation (RAG).

## Pipeline Architecture

1. Language Detection: Identifies the user's language (Arabic/English) to ensure response alignment.
2. Sentiment Analysis: Detects the customer's emotional state (e.g., frustrated or neutral) to adapt the tone with appropriate empathy.
3. Intent Classification: Classifies the core intent behind the inquiry or complaint.
4. RAG & LLM Generation: Retrieves knowledge base context using FAISS and Sentence Transformers, generating the final response via the Groq LLM API.

## Tech Stack

* Backend & ML Logic: Python, Scikit-Learn, Joblib
* Deep Learning & Transformers: Hugging Face Transformers, Sentence Transformers (all-MiniLM-L6-v2)
* Vector Database: FAISS
* LLM Engine: Groq API
* Frontend & UI: Streamlit (st.chat_message, st.chat_input)
* Deployment & Storage: Streamlit Community Cloud, GitHub, Google Drive (gdown)

## Project Structure

customer-support-bot/
├── app.py
├── requirements.txt
└── README.md

## Local Setup

1. Clone the repository:
   git clone https://github.com/your-username/customer-support-bot.git
   cd customer-support-bot

2. Install dependencies:
   pip install -r requirements.txt

3. Set up your Groq API key in .streamlit/secrets.toml:
   GROQ_API_KEY = "your_groq_api_key_here"

4. Run the app locally:
   streamlit run app.py

## Deployment

1. Push app.py and requirements.txt to GitHub.
2. Upload models.zip to Google Drive with public link access and grab the File ID.
3. Set the GOOGLE_DRIVE_FILE_ID in app.py.
4. Deploy on Streamlit Community Cloud and add GROQ_API_KEY to the app Secrets.
