import streamlit as st
from transformers import pipeline
import wikipedia

# --- PAGE CONFIG ---
st.set_page_config(page_title="WikiSageQA", layout="centered")
st.title("🌐 WikiSageQA: Multilingual Wikipedia Question Answering")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    extractive = pipeline("question-answering", model="deepset/xlm-roberta-large-squad2")
    generative = pipeline("text-generation", model="gpt2")
    return extractive, generative

extractive_qa, generative_qa = load_models()

# --- GET CONTEXT ---
def get_wikipedia_content(topic, lang="en"):
    try:
        wikipedia.set_lang(lang)
        page = wikipedia.page(topic)
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        return f"❌ Disambiguation Error: Try being more specific.\nOptions: {e.options}"
    except wikipedia.exceptions.PageError:
        return f"❌ Page not found for topic: '{topic}' in language: '{lang}'"
    except Exception as e:
        return f"❌ Unexpected Error: {e}"

# --- QA FUNCTIONS ---
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

# --- HANDLE SUBMISSION ---
if submitted:
    with st.spinner("🔄 Fetching Wikipedia content and generating answer..."):
        context = get_wikipedia_content(topic, lang)

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
