import re
from typing import Optional

try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False


NEGATION_WORDS = {"not", "no", "never"}

# Drop auxiliaries for gloss output
AUXILIARY_WORDS = {
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "will", "would", "shall", "should",
    "can", "could",
    "may", "might", "must"
}


PUNCTUATION_REGEX = re.compile(r"[\.,!?;:\-\(\)\[\]\{\}\"\']+")
POSSESSIVE_S_REGEX = re.compile(r"\b(\w+)'s\b", flags=re.IGNORECASE)
TRAILING_APOSTROPHE_REGEX = re.compile(r"\b(\w+)'\b", flags=re.IGNORECASE)


def text_to_tokens(text: str) -> list[str]:
    """Normalize free-form text into lowercase word tokens for buffering/prediction."""
    if not text:
        return []

    normalized = text.replace("’", "'")
    normalized = POSSESSIVE_S_REGEX.sub(r"\1", normalized)
    normalized = TRAILING_APOSTROPHE_REGEX.sub(r"\1", normalized)
    cleaned = PUNCTUATION_REGEX.sub(" ", normalized)
    return [tok.lower() for tok in cleaned.strip().split() if tok.strip()]


def get_prediction_context_tokens(
    tokens: list[str],
    preferred_n: int = 3,
    min_n: int = 2,
) -> list[str]:
    """
    Return last tokens for prediction.
    Uses last `preferred_n` when available, otherwise last `min_n`, otherwise all.
    """
    if not tokens:
        return []

    preferred_n = max(1, preferred_n)
    min_n = max(1, min_n)
    if min_n > preferred_n:
        min_n = preferred_n

    if len(tokens) >= preferred_n:
        return tokens[-preferred_n:]
    if len(tokens) >= min_n:
        return tokens[-min_n:]
    return tokens[:]


def to_gloss(text: str, psl_index: dict | None = None) -> str:
    """
    Convert plain English text into a simple, rule-based GLOSS approximation.

    This implementation is intentionally lightweight and dependency-free. It applies
    common heuristic transformations used for English-to-gloss style:
      - Remove punctuation
      - Drop articles, copulas, and auxiliaries
      - Lift negation to NOT pre-verb if found
      - Uppercase output tokens
      - Collapse extra whitespace

    Note: This is a simplified heuristic converter and not linguistically complete.
    """

    if not text:
        return ""

    # Normalize smart quotes to straight apostrophe to simplify regex handling
    normalized = text.replace("’", "'")

    # Remove possessive suffixes before general punctuation stripping, e.g., "today's" -> "today"
    normalized = POSSESSIVE_S_REGEX.sub(r"\1", normalized)
    normalized = TRAILING_APOSTROPHE_REGEX.sub(r"\1", normalized)

    cleaned = PUNCTUATION_REGEX.sub(" ", normalized)
    tokens = cleaned.strip().split()

    gloss_tokens = []
    pending_negation = False

    # Ensure NLTK data is available (best-effort; glossing continues even if downloads fail)
    lemmatizer: Optional[WordNetLemmatizer] = None
    if _NLTK_AVAILABLE:
        try:
            # Try to find required resources; download quietly if missing
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            try:
                nltk.data.find('corpora/omw-1.4')
            except LookupError:
                nltk.download('omw-1.4', quiet=True)
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger_eng')
            except LookupError:
                # Fall back to legacy tagger name
                try:
                    nltk.data.find('taggers/averaged_perceptron_tagger')
                except LookupError:
                    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
                    # Also attempt legacy name
                    nltk.download('averaged_perceptron_tagger', quiet=True)

            lemmatizer = WordNetLemmatizer()
        except Exception:
            lemmatizer = None

    def _wn_pos(tag: str):
        if not tag:
            return wn.NOUN
        head = tag[0].upper()
        if head == 'J':
            return wn.ADJ
        if head == 'V':
            return wn.VERB
        if head == 'N':
            return wn.NOUN
        if head == 'R':
            return wn.ADV
        return wn.NOUN

    # POS tag the whole sequence for better context-aware tags
    tagged_tokens = []
    if lemmatizer is not None:
        try:
            tagged_tokens = pos_tag(tokens)
        except Exception:
            tagged_tokens = [(t, 'NN') for t in tokens]
    else:
        tagged_tokens = [(t, 'NN') for t in tokens]

    for token, pos in tagged_tokens:
        lower = token.lower()

        if lower in NEGATION_WORDS:
            pending_negation = True
            continue

        if pending_negation:
            gloss_tokens.append("NOT")
            pending_negation = False

        # Drop auxiliaries (after handling negation)
        if lower in AUXILIARY_WORDS:
            continue

        # NLTK lemmatization (with POS) if available; otherwise keep as-is
        lemma = lower
        if lemmatizer is not None:
            try:
                wn_pos = _wn_pos(pos)
                lemma = lemmatizer.lemmatize(lower, pos=wn_pos)
                # If tagged as NOUN but token looks like a verb form, try VERB
                if wn_pos == wn.NOUN and (lower.endswith("ed") or lower.endswith("ing")):
                    v_lemma = lemmatizer.lemmatize(lower, pos=wn.VERB)
                    if v_lemma != lower:
                        lemma = v_lemma
            except Exception:
                try:
                    # Fallback: try verb, then noun
                    lemma = lemmatizer.lemmatize(lower, pos=wn.VERB)
                    if lemma == lower:
                        lemma = lemmatizer.lemmatize(lower, pos=wn.NOUN)
                except Exception:
                    lemma = lower

        # Prefer PSL canonical form for the lemma if provided; otherwise uppercase lemma
        if psl_index is not None:
            by_lower = psl_index.get("by_lower", {})
            if lemma in by_lower:
                gloss_tokens.append(by_lower[lemma][0])
            else:
                gloss_tokens.append(lemma.upper())
        else:
            gloss_tokens.append(lemma.upper())

    # If negation was trailing with no following content, keep it
    if pending_negation:
        gloss_tokens.append("NOT")

    return " ".join(gloss_tokens)


__all__ = ["to_gloss", "text_to_tokens", "get_prediction_context_tokens"]
