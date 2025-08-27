"""Offline tests for Weaviate schema diff logic (no network required)."""
from shared.vector.weaviate_schema import diff_schema


def test_diff_schema_add_class_and_properties():
    current = []
    desired = [
        {
            "class": "Instrument",
            "description": "Trading instrument",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "symbol", "dataType": ["text"], "description": "Ticker"},
                {"name": "sector", "dataType": ["text"], "description": "Sector"},
            ],
        }
    ]
    d = diff_schema(current, desired)
    assert len(d["add_classes"]) == 1
    assert d["add_properties"] == {}
    assert d["remove_classes"] == []


def test_diff_schema_add_properties_only():
    current = [
        {
            "class": "NewsArticle",
            "description": "News",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "title", "dataType": ["text"]},
            ],
        }
    ]
    desired = [
        {
            "class": "NewsArticle",
            "description": "News",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "title", "dataType": ["text"]},
                {"name": "body", "dataType": ["text"]},
            ],
        }
    ]
    d = diff_schema(current, desired)
    assert d["add_classes"] == []
    assert "NewsArticle" in d["add_properties"]
    assert d["remove_classes"] == []


def test_diff_schema_modify_and_remove():
    current = [
        {
            "class": "OldClass",
            "description": "Deprecated",
            "vectorizer": "text2vec-transformers",
            "properties": []
        },
        {
            "class": "NewsArticle",
            "description": "Old desc",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "title", "dataType": ["text"]},
            ],
        }
    ]
    desired = [
        {
            "class": "NewsArticle",
            "description": "Updated desc",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "title", "dataType": ["text"]},
                {"name": "body", "dataType": ["text"]},
            ],
        }
    ]
    d = diff_schema(current, desired)
    assert d["remove_classes"] == ["OldClass"]
    assert any(mod[0] == "NewsArticle" for mod in d["modify_classes"])
    assert "NewsArticle" in d["add_properties"]
