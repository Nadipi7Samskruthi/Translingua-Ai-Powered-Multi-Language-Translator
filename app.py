import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import random

# Available language pairs for translation
LANGUAGE_PAIRS = {
    "English to French": ("en", "fr"),
    "French to English": ("fr", "en"),
    "English to German": ("en", "de"),
    "German to English": ("de", "en"),
    "English to Spanish": ("en", "es"),
    "Spanish to English": ("es", "en"),
}

@st.cache_resource  # Cache model loading for efficiency
def load_model(src_lang, tgt_lang):
    """Load the MarianMT model and tokenizer for translation"""
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text, src_lang, tgt_lang):
    """Translate the given text from source to target language"""
    tokenizer, model = load_model(src_lang, tgt_lang)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Streamlit UI
st.title("TransLingua: AI-Powered Multi-Language Translator 🌍")
st.write("Translate text seamlessly between multiple languages using AI.")

# Sidebar Navigation
option = st.sidebar.selectbox("Choose an option:", ["Translation", "Note Taking", "Word Guessing Game"])

# ------------------------------------------
# 1️⃣ Language Translation Section
# ------------------------------------------
if option == "Translation":
    st.subheader("Translate Text")

    text_input = st.text_area("Enter text to translate:", height=150)
    selected_pair = st.selectbox("Select Language Pair:", list(LANGUAGE_PAIRS.keys()))

    if st.button("Translate"):
        if text_input.strip():
            src, tgt = LANGUAGE_PAIRS[selected_pair]
            translation = translate_text(text_input, src, tgt)
            st.success("Translated Text:")
            st.write(translation)
        else:
            st.warning("Please enter text to translate.")

# ------------------------------------------
# 2️⃣ Note-Taking Section
# ------------------------------------------
elif option == "Note Taking":
    st.subheader("Notes")
    note = st.text_area("Write your notes here:", height=150)

    if "notes" not in st.session_state:
        st.session_state.notes = []

    if st.button("Save Note"):
        if note.strip():
            st.session_state.notes.append(note)
            st.success("Note saved!")
        else:
            st.warning("Cannot save an empty note.")

    # Display saved notes
    if st.session_state.notes:
        st.subheader("Saved Notes")
        for idx, saved_note in enumerate(st.session_state.notes):
            st.write(f"{idx + 1}. {saved_note}")

# ------------------------------------------
# 3️⃣ Word Guessing Game (English to French)
# ------------------------------------------
elif option == "Word Guessing Game":
    st.subheader("Word Guessing Game 🎮")
    st.markdown("### **Translation from English to French**")  # New heading

    # Sample words dataset (without descriptions)
    words = {
        "hello": "bonjour",
        "goodbye": "au revoir",
        "thank you": "merci",
        "please": "s'il vous plaît",
        "love": "amour",
        "friend": "ami",
        "family": "famille",
        "happy": "heureux",
        "sad": "triste",
        "food": "nourriture",
        "water": "eau",
        "book": "livre",
        "music": "musique",
        "sun": "soleil",
        "moon": "lune"
    }

    # Initialize session state for tracking game progress
    if "current_word" not in st.session_state:
        st.session_state.current_word = random.choice(list(words.keys()))
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "checked" not in st.session_state:
        st.session_state.checked = False

    # Get the current word and its correct translation
    current_word = st.session_state.current_word
    correct_translation = words[current_word]

    # Display the word for translation
    st.write(f"Translate this word into **French**: **{current_word}**")

    # Input field
    user_input = st.text_input("Your answer:", value=st.session_state.user_input)

    # Check button logic
    if st.button("Check"):
        st.session_state.user_input = user_input  # Store input
        if user_input.lower().strip() == correct_translation.lower():
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Incorrect! The correct answer is: **{correct_translation}**")
        st.session_state.checked = True  # Mark as checked

    # "Next" button appears only after checking
    if st.session_state.checked:
        if st.button("Next"):
            st.session_state.current_word = random.choice(list(words.keys()))  # Pick new word
            st.session_state.user_input = ""  # Reset input
            st.session_state.checked = False  # Reset check state
            st.rerun()  # Rerun app to refresh state
