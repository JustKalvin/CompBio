import streamlit as st
from transformers import pipeline
import requests
import re
import os
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

# Load NER model
ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all")

# Set OpenRouter API Key (bisa juga dari st.secrets["OPENROUTER_API_KEY"])
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")  # Ganti jika ingin hardcode
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# Fungsi panggil OpenRouter
def explain_entity(term):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app-name.streamlit.app",  # opsional, ganti dengan URL app kamu
    }
    body = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": f"Explain the medical term: {term} in clear and concise."}
        ],
        "max_tokens": 600
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        last_period_index = content.rfind('.')
        if last_period_index != -1:
            content = content[:last_period_index + 1]
        return content
    else:
        return "❌ Error from OpenRouter API: " + response.text

# Daftar stopwords
stopwords = [
    "in", "the", "a", "an", "and", "of", "to", "with", "is", "was", "were", "are",
    "on", "at", "by", "for", "from", "as", "that", "this", "it", "its", "which",
    "who", "whom", "whose", "where", "when", "how", "what", "why", "or", "but",
    "not", "so", "such", "than", "then", "there", "thus", "very", "can", "could",
    "should", "would", "will", "may", "might", "must", "shall", "do", "does", "did",
    "done", "having", "have", "has", "had", "be", "being", "been", "about", "above",
    "below", "between", "into", "through", "during", "before", "after", "over",
    "under", "again", "further", "also", "because", "therefore", "however", "yet",
    "if", "unless", "until", "beside", "besides", "whether", "each", "every",
    "either", "neither", "both", "some", "any", "other", "another", "few", "many",
    "several", "all", "most", "more", "less", "least", "own", "same", "different",
    "someone", "somebody", "anyone", "anybody", "everyone", "everybody",
    "no one", "nobody", "nothing", "everything", "anything", "something", "him",
    "her", "his", "hers", "their", "theirs", "our", "ours", "my", "mine", "your",
    "yours", "its", "whenever", "wherever", "whatever", "whichever", "whoever",
    "whomever", "seems", "appears", "approximately", "likely", "nearly", "about",
    "almost", "probably", "possibly", "perhaps", "sometimes", "often", "always",
    "never", "rarely", "usually", "normally", "generally", "mostly", "commonly",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've",
    "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", "should", "should've",
    "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn",
    "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't", "mustn",
    "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't"
]

# Fungsi untuk mengekstrak teks dari PDF
def extract_text_from_pdf(uploaded_file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=None)
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    text = ""
    for page in PDFPage.get_pages(uploaded_file):
        page_interpreter.process_page(page)
        text += fake_file_handle.getvalue()
        fake_file_handle.truncate(0)
        fake_file_handle.seek(0)
    converter.close()
    return text

# Ekstraksi entitas medis
def extract_entities(text):
    entities = ner_pipeline(text)
    merged_entities = []
    current_word = ""
    current_entity = None
    seen_entities = set()  # Set untuk melacak entitas yang sudah dikenali
    for ent in entities:
        word = ent["word"]
        entity_type = ent["entity"].split("-")[-1]
        # Gabungkan kata yang terpecah
        if word.startswith("#"):
            current_word += word.lstrip("#")
        else:
            # Cek apakah entitas sudah ada sebelumnya
            if current_word and current_entity and current_word.lower() not in stopwords and current_word not in seen_entities:
                merged_entities.append({"word": current_word, "entity": current_entity})
                seen_entities.add(current_word.lower())  # Tambahkan entitas ke set
            current_word = word
            current_entity = entity_type
    # Menambahkan entitas terakhir jika tidak ada duplikat
    if current_word and current_entity and current_word.lower() not in stopwords and current_word not in seen_entities:
        merged_entities.append({"word": current_word, "entity": current_entity})
        seen_entities.add(current_word.lower())
    return merged_entities

# Highlight teks dan tampilkan entitas
def highlight_text(text, entities):
    words = text.split(' ')
    entity_words = {ent["word"].lower(): ent for ent in entities}
    for i, word in enumerate(words):
        clean_word = re.sub(r"[^\w\s]", "", word)
        if clean_word.lower() in entity_words:
            words[i] = f"<span style='background-color: #ffcc80; color: black; padding: 2px; border-radius: 4px;'>{word}</span>"
    highlighted_text = ' '.join(words)
    entity_list = "<ul>"
    for ent in entities:
        entity_list += f"<li><strong>{ent['word']}</strong> ({ent['entity']})</li>"
    entity_list += "</ul>" if entities else "<p><em>No medical entities detected.</em></p>"
    return highlighted_text, entity_list
    
def display_entities_horizontally(entities, columns_per_row=5):
    num_entities = len(entities)
    num_rows = (num_entities + columns_per_row - 1) // columns_per_row
    for i in range(num_rows):
        cols = st.columns(columns_per_row)
        for j in range(columns_per_row):
            index = i * columns_per_row + j
            if index < num_entities:
                ent = entities[index]
                cols[j].markdown(f"- **{ent['word']}** ({ent['entity']})")
                
# --- Streamlit UI ---
st.set_page_config(page_title="Synth", layout="wide")
st.markdown("<h1 style='text-align: center;'>🩺 Sȳnth</h1><p style='text-align: center;'>💬 Get ready with Sȳnth!</p>", unsafe_allow_html=True)

# Upload PDF File
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

extracted_text = ""
if uploaded_file is not None:
    st.subheader("📄 PDF Content:")
    text_content = extract_text_from_pdf(uploaded_file)
    extracted_text = text_content
    
chat_input = st.text_input("Your Message", value = extracted_text, placeholder="Type your message here...")
if chat_input:
    st.subheader("⚕️ Medical Entity Recognition (Chat Input)")
    preprocessed_msg = re.sub(r"[^a-zA-Z0-9\s]", "", chat_input).lower()
    entities_chat = extract_entities(preprocessed_msg)
    highlighted_html_chat, entity_html_list_chat = highlight_text(chat_input, entities_chat)
    st.markdown("<h3>🔍 Highlighted Text</h3>", unsafe_allow_html=True)
    st.markdown(highlighted_html_chat, unsafe_allow_html=True)
    st.markdown("<h4>🔍 Recognized Medical Entities:</h4>", unsafe_allow_html=True)
    display_entities_horizontally(entities_chat)

    unique_terms_chat = list({ent["word"] for ent in entities_chat})
    if unique_terms_chat:
        st.markdown("### 🧠 Pilih entitas dari pesan untuk penjelasan:")
        selected_entities_chat = []
        num_entities_chat = len(entities_chat)
        cols_chat = st.columns(5)
        for i, ent in enumerate(entities_chat):
            with cols_chat[i % 5]:
                if st.checkbox(f"{ent['word']} ({ent['entity']})", key=f"chat_entity_{ent['word']}"):
                    selected_entities_chat.append(ent["word"])
        if st.button("Send (Entities)"):
            if selected_entities_chat:
                combined_term_chat = ' '.join(selected_entities_chat)
                explanation_chat = explain_entity(combined_term_chat)
                st.markdown(f"**ℹ️ Explanation for '{combined_term_chat}':**")
                st.info(explanation_chat)
            else:
                st.warning("Silakan pilih minimal satu entitas sebelum mengirim.")
