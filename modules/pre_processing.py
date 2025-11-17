import re
import numpy as np
from math import log2
from collections import Counter
import emoji
import spacy

nlp = spacy.load("en_core_web_sm")

SELECTED_POS = {
    "NOUN": "pos_noun_ratio",
    "PUNCT": "pos_punct_ratio",
    "DET": "pos_det_ratio",
    "ADJ": "pos_adj_ratio",
    "ADV": "pos_adv_ratio",
    "VERB": "pos_verb_ratio",
    "PRON": "pos_pron_ratio"
}


# --------------------------
# --- HELPER FUNCTIONS -----
# --------------------------

def clean_text(text: str) -> str:
    """Cleans text while preserving punctuation."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalized_token_entropy(words):
    n = len(words)
    if n <= 1:
        return 0.0
    freq = Counter(words)
    probs = [v / n for v in freq.values()]
    H = -sum(p * log2(p) for p in probs)
    return H / log2(n)


def calculate_sentence_token_entropy(sentence: str):
    doc = nlp(sentence)
    words = [token.text for token in doc]
    return normalized_token_entropy(words)


def extract_text_features(text: str):
    doc = nlp(text)
    words = [token.text for token in doc if token.is_alpha]
    num_words = len(words)
    unique_words = len(set(words))
    denom = max(1, num_words)

    # POS ratios
    pos_counts = {v: 0 for v in SELECTED_POS.values()}
    for token in doc:
        if token.pos_ in SELECTED_POS:
            pos_counts[SELECTED_POS[token.pos_]] += 1
    for k in pos_counts:
        pos_counts[k] /= denom

    # punctuation ratio
    punct_ratio = sum(token.pos_ == "PUNCT" for token in doc) / denom

    base_features = {
        "punct_ratio": punct_ratio,
        "avg_token_length": np.mean([len(w) for w in words]) if words else 0,
        "type_token_ratio": unique_words / denom,
        "hapax_legomena_ratio": sum(1 for w in set(words) if words.count(w) == 1) / denom
    }

    return {**base_features, **pos_counts}


# -----------------------------------
# 💡 MAIN FUNCTION (Final Preprocess)
# -----------------------------------
def preprocess_essay(text: str) -> dict:
    """
    Full preprocessing pipeline:
    - Clean text
    - Extract lexical + POS features
    - Compute entropy (global + sentence)
    - Return dict of features + cleaned text
    """
    cleaned = clean_text(text)
    doc = nlp(cleaned)

    # sentence-level entropy
    sentence_entropies = [
        calculate_sentence_token_entropy(sent.text)
        for sent in doc.sents
    ]
    avg_sentence_entropy = np.mean(sentence_entropies) if sentence_entropies else 0

    # word list for global entropy
    words = [token.text for token in doc if token.is_alpha]
    global_entropy = normalized_token_entropy(words)

    # main features
    feature_dict = extract_text_features(cleaned)

    # add entropies
    feature_dict["sentence_token_entropy"] = avg_sentence_entropy
    feature_dict["token_entropy"] = global_entropy
    feature_dict["n_tokens"] = len(words)

    # include cleaned text (useful for TF-IDF)
    feature_dict["cleaned_text"] = cleaned

    return feature_dict
