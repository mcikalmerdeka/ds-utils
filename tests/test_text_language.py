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
        translation_utils, "_get_identifier", lambda *args, **kwargs: DummyIdentifier()
    )
    df = pd.DataFrame({"text": ["Hola amigo", "Hello world"]})

    target, other = translation_utils.detect_language_rows(
        df, ["text"], target_langs=None, use_parallel=False
    )

    assert target["text"].tolist() == ["Hola amigo"]
    assert other["text"].tolist() == ["Hello world"]


def test_detect_language_rows_all_null_does_not_require_extras():
    df = pd.DataFrame({"text": [None, None]})

    target, other = translation_utils.detect_language_rows(
        df, ["text"], target_langs=None, use_parallel=False
    )

    assert target.empty
    assert other.equals(df)


class FakeLanguageIdentifier:
    def __init__(self, languages=None):
        self.languages = list(languages or ["en", "es", "fr"])

    @classmethod
    def from_modelstring(cls, model, norm_probs=True):
        return cls()

    def set_languages(self, languages):
        self.languages = list(languages)


def test_get_identifier_does_not_leak_explicit_language_constraints(monkeypatch):
    monkeypatch.setattr(translation_utils, "LANGID_AVAILABLE", True)
    monkeypatch.setattr(
        translation_utils, "LanguageIdentifier", FakeLanguageIdentifier
    )
    monkeypatch.setattr(translation_utils, "model", object())
    monkeypatch.setattr(translation_utils, "_identifier", None, raising=False)
    monkeypatch.setattr(translation_utils, "_identifier_all", None, raising=False)

    constrained = translation_utils._get_identifier({"hi"})
    full = translation_utils._get_identifier(None)

    assert constrained is not full
    assert set(constrained.languages) == {"hi", "en"}
    assert "es" in full.languages
