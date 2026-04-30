from __future__ import annotations

from src.preprocessing.parser import parse_request


def test_parse_request_basic():
    entry = {
        "method": "GET",
        "path": "/api/users/123/profile",
        "query": "id=123&token=abc",
        "user_agent": "UA",
        "status": 200,
        "response_size": 42,
    }
    pr = parse_request(entry)
    assert pr.method == "GET"
    assert pr.path == "/api/users/123/profile"
    assert pr.query_params["id"] == "123"
    assert pr.headers["User-Agent"] == "UA"


