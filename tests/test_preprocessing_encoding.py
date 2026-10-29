import pandas as pd
import pytest

from utils.preprocessing import feature_encoding


def test_feature_encoding_preserves_unlisted_object_columns():
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a"],
            "free_text": ["hello", "world", "foo"],
            "num": [1, 2, 3],
        }
    )

    encoded, _ = feature_encoding(df, nominal_columns=["cat"])

    assert encoded["free_text"].tolist() == df["free_text"].tolist()


def test_feature_encoding_raises_on_unseen_binary_category_by_default():
    train = pd.DataFrame({"b": ["yes", "no", "yes"]})
    test = pd.DataFrame({"b": ["yes", "maybe"]})
    _, encoders = feature_encoding(train, binary_columns=["b"])

    with pytest.raises(ValueError, match="maybe"):
        feature_encoding(test, binary_columns=["b"], encoders=encoders)


def test_feature_encoding_ignore_maps_unseen_binary_to_zero():
    train = pd.DataFrame({"b": ["yes", "no", "yes"]})
    test = pd.DataFrame({"b": ["yes", "maybe"]})
    _, encoders = feature_encoding(train, binary_columns=["b"])

    encoded, _ = feature_encoding(
        test,
        binary_columns=["b"],
        encoders=encoders,
        handle_unknown="ignore",
    )

    assert encoded["b"].tolist() == [1.0, 0.0]


def test_feature_encoding_rejects_use_encoded_value_for_nominal():
    df = pd.DataFrame({"cat": ["a", "b", "a"]})

    with pytest.raises(ValueError, match="use_encoded_value"):
        feature_encoding(
            df,
            nominal_columns=["cat"],
            handle_unknown="use_encoded_value",
        )
