import pandas as pd
import pytest

from utils import text_preprocessing, translation_utils
from utils.text_preprocessing import TextPreprocessor


class DummyIdentifier:
    def classify(self, text):
        if "Hola" in text:
            return "es", 0.99
        return "en", 0.99


def test_detection_requires_only_langid(monkeypatch):
    def fake_detection_check():
        return None

    def fail_translation_check():
        raise ImportError("deep-translator is missing")

    monkeypatch.setattr(
        translation_utils, "_require_detection_deps", fake_detection_check
    )
    monkeypatch.setattr(
        translation_utils, "_require_translation_deps", fail_translation_check
    )
    monkeypatch.setattr(
        translation_utils, "_get_identifier", lambda *args, **kwargs: DummyIdentifier()
    )
    data = pd.DataFrame({"text": ["Hola amigo", "Hello world"]})

    target, other = translation_utils.detect_language_rows(
        data, ["text"], target_langs=None, use_parallel=False
    )

    assert target["text"].tolist() == ["Hola amigo"]
    assert other["text"].tolist() == ["Hello world"]


def test_translation_requires_only_deep_translator(monkeypatch):
    monkeypatch.setattr(translation_utils, "DEEP_TRANSLATOR_AVAILABLE", True)
    monkeypatch.setattr(translation_utils, "LANGID_AVAILABLE", False)
    monkeypatch.setattr(
        translation_utils,
        "_translate_batch_with_retry",
        lambda texts, **kwargs: list(texts),
    )

    result = translation_utils.translate_series(
        pd.Series(["hola", "mundo"]), inter_batch_delay=0
    )

    assert result.tolist() == ["hola", "mundo"]


def test_detection_check_names_only_langid(monkeypatch):
    monkeypatch.setattr(translation_utils, "LANGID_AVAILABLE", False)
    monkeypatch.setattr(translation_utils, "DEEP_TRANSLATOR_AVAILABLE", False)

    with pytest.raises(ImportError, match="langid"):
        translation_utils._require_detection_deps()


def test_translation_check_names_only_deep_translator(monkeypatch):
    monkeypatch.setattr(translation_utils, "LANGID_AVAILABLE", False)
    monkeypatch.setattr(translation_utils, "DEEP_TRANSLATOR_AVAILABLE", False)

    with pytest.raises(ImportError, match="deep-translator"):
        translation_utils._require_translation_deps()


def test_nltk_missing_message_mentions_extra(monkeypatch):
    monkeypatch.setattr(text_preprocessing, "NLTK_AVAILABLE", False)
    preprocessor = TextPreprocessor()

    with pytest.raises(ImportError) as exc_info:
        preprocessor.process_text("hello world")

    message = str(exc_info.value)
    assert "uv sync --extra text" in message
    assert "nltk.download" in message
