# 🌐 WikiSageQA: Multilingual Wikipedia Question Answering

WikiSageQA is an interactive Streamlit app that lets users ask questions about any topic using real-time Wikipedia content. It supports **multilingual queries** and allows you to choose between:

- **Extractive QA (DistilBERT)** for precise answers
- **Generative QA (GPT-2)** for creative responses

## 🚀 Features

- Wikipedia search with language selection
- Support for English, Hindi, French, Spanish, and German
- Two QA modes: extractive and generative
- Real-time answer generation using HuggingFace models


## 🔧 How to Run

```bash
# Clone the repository
git clone https://github.com/IshaanMe/WikiSageQA.git
cd WikiSageQA

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
