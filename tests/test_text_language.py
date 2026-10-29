import pandas as pd

from utils import translation_utils
from utils.text_preprocessing import TextPreprocessor
from utils.translation_utils import _is_target_language


class DummyIdentifier:
    def __init__(self):
        self.languages = None

    def set_languages(self, languages):
        self.languages = languages

    def classify(self, text):
        if "Hola" in text or "Bonjour" in text:
            return "es" if "Hola" in text else "fr", 0.99
        return "en", 0.99


def test_remove_urls_preserves_decimals_money_and_ips():
    preprocessor = TextPreprocessor()

    assert preprocessor.remove_urls("Version 3.14 costs $2.50") == "Version 3.14 costs $2.50"
    assert preprocessor.remove_urls("Server 192.168.1.1 is down") == "Server 192.168.1.1 is down"


def test_remove_urls_removes_real_urls():
    preprocessor = TextPreprocessor()

    result = preprocessor.remove_urls("Visit https://example.com and www.example.org")

    assert "example.com" not in result
    assert "example.org" not in result


def test_process_text_removes_email_without_leaving_prefix():
    preprocessor = TextPreprocessor()

    result = preprocessor.process_text(
        "Contact test@example.com",
        options={"remove_stopwords": False, "lemmatize": False},
    )

    assert "test@" not in result
    assert "@" not in result


def test_short_spanish_text_is_detected_with_explicit_target():
    assert _is_target_language("Hola amigo", {"es"}, DummyIdentifier()) is True


def test_empty_target_set_detects_non_english():
    assert _is_target_language("Bonjour tout le monde", set(), DummyIdentifier()) is True
    assert _is_target_language("Hello world", set(), DummyIdentifier()) is False


def test_detect_language_rows_target_none_finds_non_english(monkeypatch):
    monkeypatch.setattr(translation_utils, "_require_translate_deps", lambda: None)
    monkeypatch.setattr(
        translation_utils, "_get_identifier", lambda: DummyIdentifier()
    )
    df = pd.DataFrame({"text": ["Hola amigo", "Hello world"]})

    target, other = translation_utils.detect_language_rows(
        df, ["text"], target_langs=None, use_parallel=False
    )

    assert target["text"].tolist() == ["Hola amigo"]
    assert other["text"].tolist() == ["Hello world"]
