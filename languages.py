"""
Language management module.
Maps (source_language, target_language) pairs to their pretrained MarianMT models.
"""

# List of supported languages shown in the UI dropdowns
LANGUAGES = ["English", "Hindi", "French", "Spanish", "German"]

# Mapping: (source, target) -> Hugging Face model name
# Each pair uses a dedicated pretrained Neural Machine Translation model.
MODELS = {
    # English -> others
    ("English", "Hindi"):   "Helsinki-NLP/opus-mt-en-hi",
    ("English", "French"):  "Helsinki-NLP/opus-mt-en-fr",
    ("English", "Spanish"): "Helsinki-NLP/opus-mt-en-es",
    ("English", "German"):  "Helsinki-NLP/opus-mt-en-de",

    # Others -> English
    ("Hindi", "English"):   "Helsinki-NLP/opus-mt-hi-en",
    ("French", "English"):  "Helsinki-NLP/opus-mt-fr-en",
    ("Spanish", "English"): "Helsinki-NLP/opus-mt-es-en",
    ("German", "English"):  "Helsinki-NLP/opus-mt-de-en",
}


def get_model_name(source_lang: str, target_lang: str):
    """
    Return the pretrained model name for the given language pair.
    Returns None if the pair is not supported.
    """
    return MODELS.get((source_lang, target_lang))


def supported_pairs():
    """Return a list of all supported (source, target) tuples."""
    return list(MODELS.keys())