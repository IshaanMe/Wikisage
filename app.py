
import streamlit as st
from transformers import pipeline
import wikipediaapi

# --- PAGE CONFIG ---
st.set_page_config(page_title="WikiSageQA", layout="centered")
st.title("🌐 WikiSageQA: Multilingual Wikipedia Question Answering")

# --- LOAD WIKIPEDIA ---
def get_wikipedia_full_page(topic, lang="en"):
    wiki = wikipediaapi.Wikipedia(lang)
    page = wiki.page(topic)
    if page.exists():
        return page.text
    return f"❌ Error: Page '{topic}' not found in Wikipedia ({lang})"

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    extractive = pipeline("question-answering", model="deepset/xlm-roberta-large-squad2")
    generative = pipeline("text-generation", model="gpt2")
    return extractive, generative

extractive_qa, generative_qa = load_models()

# --- GET ANSWERS ---
def get_extractive_answer(question, context):
    try:
        result = extractive_qa(question=question, context=context)
        return result["answer"]
    except Exception as e:
        return f"⚠️ Extractive QA Error: {e}"

def get_generative_answer(question, context):
    try:
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        outputs = generative_qa(prompt, max_new_tokens=100, do_sample=True, temperature=0.8)
        return outputs[0]["generated_text"].split("Answer:")[-1].strip()
    except Exception as e:
        return f"⚠️ Generative QA Error: {e}"

# --- UI FORM ---
with st.form("qa_form"):
    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("🔍 Wikipedia Topic", value="Albert Einstein")
    with col2:
        lang = st.selectbox("🌐 Language", options=["en", "hi", "fr", "es", "de"], index=0)

    question = st.text_input("❓ Your Question", value="What is he famous for?")
    mode = st.radio("⚙️ QA Mode", ["Extractive (mBERT)", "Generative (GPT-2)"], horizontal=True)
    submitted = st.form_submit_button("Get Answer")

# --- HANDLE FORM SUBMISSION ---
if submitted:
    with st.spinner("🔄 Getting answer..."):
        context = get_wikipedia_full_page(topic, lang)

        if context.startswith("❌"):
            st.error(context)
        else:
            st.markdown("### 📘 Wikipedia Context:")
            st.write(context[:2000] + "..." if len(context) > 2000 else context)

            if mode.startswith("Extractive"):
                answer = get_extractive_answer(question, context)
                st.markdown("### 🧠 Extractive Answer (mBERT):")
            else:
                answer = get_generative_answer(question, context)
                st.markdown("### 🧠 Generative Answer (GPT-2):")

            st.success(answer)
