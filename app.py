"""
Multilingual Translator Using Natural Language Processing
---------------------------------------------------------
A simple Streamlit app that translates text between multiple languages
using pretrained Transformer-based Neural Machine Translation (NMT) models.

Workflow:
    User Input -> Language Selection -> Preprocessing -> Tokenization
    -> Pretrained NMT Model -> Sequence Generation -> Token Decoding
    -> Translated Text
"""

import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

from languages import LANGUAGES, get_model_name


# ---------------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Multilingual Translator",
    page_icon="🌐",
    layout="centered",
)


# ---------------------------------------------------------------
# 2. Model Loading (cached so it loads only once)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """
    Load the MarianMT tokenizer and model for a given model name.
    Cached by Streamlit so the model isn't reloaded on every click.
    """
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


# ---------------------------------------------------------------
# 3. Core Translation Function
# ---------------------------------------------------------------
def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text from source_lang to target_lang using MarianMT.

    Steps:
        1. Get the correct model name for the language pair.
        2. Tokenize the input text (text -> token IDs).
        3. Run the model to generate translated token IDs.
        4. Decode the generated tokens back to human-readable text.
    """
    model_name = get_model_name(source_lang, target_lang)
    if model_name is None:
        return "⚠️ This language pair is not supported."

    # Load (or fetch from cache) the model + tokenizer
    tokenizer, model = load_model(model_name)

    # --- Tokenization ---
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # --- Sequence Generation ---
    translated_tokens = model.generate(**inputs)

    # --- Token Decoding ---
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text


# ---------------------------------------------------------------
# 4. Session State for swap / clear functionality
# ---------------------------------------------------------------
if "source_lang" not in st.session_state:
    st.session_state.source_lang = "English"
if "target_lang" not in st.session_state:
    st.session_state.target_lang = "Hindi"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""


def swap_languages():
    """Swap source and target languages."""
    st.session_state.source_lang, st.session_state.target_lang = (
        st.session_state.target_lang,
        st.session_state.source_lang,
    )


def clear_all():
    """Clear input and output text boxes."""
    st.session_state.input_text = ""
    st.session_state.output_text = ""


# ---------------------------------------------------------------
# 5. User Interface
# ---------------------------------------------------------------
st.title("🌐 Multilingual Translator")
st.caption("Translate text between multiple languages using pretrained Neural Machine Translation models.")

st.divider()

# --- Language selection row ---
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    source_lang = st.selectbox(
        "Source Language",
        LANGUAGES,
        index=LANGUAGES.index(st.session_state.source_lang),
        key="source_select",
    )
    st.session_state.source_lang = source_lang

with col2:
    st.write("")  # spacing
    st.write("")  # spacing
    st.button("🔁 Swap", on_click=swap_languages, use_container_width=True)

with col3:
    target_lang = st.selectbox(
        "Target Language",
        LANGUAGES,
        index=LANGUAGES.index(st.session_state.target_lang),
        key="target_select",
    )
    st.session_state.target_lang = target_lang

# --- Text input ---
input_text = st.text_area(
    "Enter text to translate",
    value=st.session_state.input_text,
    height=150,
    placeholder="Type or paste your text here...",
)

# --- Buttons ---
btn_col1, btn_col2 = st.columns([1, 1])
with btn_col1:
    translate_clicked = st.button("🚀 Translate", type="primary", use_container_width=True)
with btn_col2:
    st.button("🗑️ Clear", on_click=clear_all, use_container_width=True)

# --- Translation logic ---
if translate_clicked:
    if not input_text.strip():
        st.warning("Please enter some text to translate.")
    elif source_lang == target_lang:
        st.info("Source and target languages are the same.")
        st.session_state.output_text = input_text
    else:
        with st.spinner("Translating..."):
            try:
                result = translate(input_text.strip(), source_lang, target_lang)
                st.session_state.output_text = result
            except Exception as e:
                st.error(f"Translation failed: {e}")
                st.session_state.output_text = ""

# --- Output ---
if st.session_state.output_text:
    st.divider()
    st.subheader("📝 Translated Text")
    st.success(st.session_state.output_text)

# --- Footer ---
st.divider()
st.caption("Built with Python • Streamlit • Hugging Face Transformers • MarianMT")
