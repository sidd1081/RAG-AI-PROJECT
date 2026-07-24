from app.preprocessing.sentence_splitter import split_sentences


def test_splits_on_sentence_boundaries():
    text = "This is one sentence. This is another one! Is this a third?"
    result = split_sentences(text)
    assert result == [
        "This is one sentence.",
        "This is another one!",
        "Is this a third?",
    ]


def test_empty_input_returns_empty_list():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_single_sentence_no_split():
    assert split_sentences("Just one sentence here.") == ["Just one sentence here."]
