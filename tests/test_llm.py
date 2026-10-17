"""How a model is asked to think, what a failure says, and what the live model
list offers each role — against a scripted provider, never the network."""

import asyncio
import json

import httpx
import pytest

from home_core import llm
from home_core.llm import THINKS, Role, claude_options, failure_text


def _provider(monkeypatch, handler):
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(llm, "_client", lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(record)))
    return seen


def test_sonnet_5_5_turns_thinking_off_between_tools():
    assert claude_options("claude-sonnet-5-5", think=False) == {"thinking": {"type": "between_tools"}}


def test_opus_5_5_cannot_turn_thinking_off_so_it_thinks_least():
    opts = claude_options("claude-opus-5-5", think=False)
    assert "thinking" not in opts
    assert opts == {"output_config": {"effort": "low"}}


def test_a_model_the_table_does_not_know_is_sent_neither_field():
    assert claude_options("claude-next-1", think=False) == {}


def test_thinking_on_is_adaptive_at_the_asked_effort():
    assert claude_options("claude-sonnet-5-5", think=True, effort="medium") == {
        "thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"},
    }


def test_a_schema_joins_the_effort_instead_of_replacing_it():
    opts = claude_options("claude-opus-5-5", think=False, schema={"type": "object"})
    assert opts["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert claude_options("claude-opus-5-5", think=False) == {"output_config": {"effort": "low"}}


@pytest.mark.parametrize(("status", "words"), [
    (401, "rejected the API key"),
    (404, "has no such model"),
    (429, "rate limit"),
    (529, "unavailable right now"),
    (None, "cannot be reached"),
    (400, "refused the request (400: bad"),
])
def test_every_failure_is_one_sentence_naming_the_provider(status, words):
    text = failure_text("anthropic", status, "bad")
    assert text.startswith("Anthropic ")
    assert words in text


def test_the_key_is_masked_to_its_ends():
    assert llm.mask_key("sk-ant-api03-abcdefghijkl-wxyz") == "sk-a…wxyz"
    assert llm.mask_key("short") == "•••••"
    assert llm.mask_key(None) == ""


def test_the_model_list_carries_capabilities_and_each_role_gets_what_it_needs(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={"has_more": False, "data": [
            {"id": "claude-sonnet-5-5", "display_name": "Claude Sonnet 5.5", "capabilities": {
                "image_input": {"supported": True},
                "structured_outputs": {"supported": True},
                "thinking": {"supported": True, "types": {"enabled": {"supported": False},
                                                          "adaptive": {"supported": True}}},
                "effort": {"supported": True, "low": {"supported": True}},
            }},
            {"id": "claude-haiku-4-5", "display_name": "Claude Haiku 4.5", "capabilities": {
                "image_input": {"supported": True},
                "thinking": {"supported": True, "types": {"enabled": {"supported": True},
                                                          "adaptive": {"supported": False}}},
                "effort": {"supported": False},
            }},
        ]})

    _provider(monkeypatch, handler)
    found = asyncio.run(llm.models("anthropic", "sk-ant-test-key"))
    sonnet, haiku = found
    assert "thinking.types.adaptive" in sonnet.caps
    assert "thinking.types.enabled" not in sonnet.caps
    chat = Role("chat", THINKS)
    image = Role("image", frozenset({"image_input"}))
    assert sonnet.serves(chat) and sonnet.serves(image)
    assert not haiku.serves(chat), "no adaptive thinking, no effort: claude_options would be refused"
    assert haiku.serves(image)


def test_a_rejected_key_lists_no_models_and_says_so(monkeypatch):
    _provider(monkeypatch, lambda r: httpx.Response(
        401, json={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}))
    with pytest.raises(llm.ProviderModelsError, match="rejected the API key"):
        asyncio.run(llm.models("anthropic", "sk-ant-wrong-key"))


def test_openai_lists_chat_models_without_capabilities_so_every_role_may_have_them():
    found = llm.openai_chat_models([
        {"id": "gpt-5", "created": 2}, {"id": "gpt-5-2025-08-07", "created": 3},
        {"id": "gpt-4o-audio-preview", "created": 4}, {"id": "o3", "created": 1},
        {"id": "text-embedding-3-large", "created": 5},
    ])
    assert [m.id for m in found] == ["gpt-5", "o3"]
    assert found[0].caps is None and found[0].serves(Role("image", frozenset({"image_input"})))


def test_the_ping_asks_the_model_to_think_least(monkeypatch):
    seen = _provider(monkeypatch, lambda r: httpx.Response(200, json={
        "model": "claude-sonnet-5-5", "stop_reason": "end_turn", "content": [{"type": "text", "text": "OK"}],
        "usage": {"input_tokens": 9, "output_tokens": 2},
    }))
    r = asyncio.run(llm.ping("anthropic", "sk-ant-test-key", "claude-sonnet-5-5"))
    assert r.ok and r.text == "OK"
    assert json.loads(seen[0].content)["thinking"] == {"type": "between_tools"}


def test_an_answer_cut_off_at_max_tokens_is_a_failure_not_a_result(monkeypatch):
    _provider(monkeypatch, lambda r: httpx.Response(200, json={
        "model": "claude-opus-5-5", "stop_reason": "max_tokens", "content": [{"type": "text", "text": "{\"zo"}],
        "usage": {"input_tokens": 9, "output_tokens": 8000},
    }))
    r = asyncio.run(llm.text_json("anthropic", "sk-ant-test-key", "claude-opus-5-5", system="s", user_text="u"))
    assert not r.ok
    assert "stop_reason=max_tokens" in r.error


def test_a_gateway_page_is_not_shown_as_the_error(monkeypatch):
    _provider(monkeypatch, lambda r: httpx.Response(502, text="<html><body>Bad gateway</body></html>"))
    r = asyncio.run(llm.ping("anthropic", "sk-ant-test-key", "claude-opus-5-5"))
    assert not r.ok
    assert "upstream gateway error" in r.error and "<html>" not in r.error


def test_the_model_list_follows_every_page(monkeypatch):
    pages = {
        None: {"data": [{"id": "claude-opus-5-5", "display_name": "Claude Opus 5.5", "capabilities": {}}],
               "has_more": True, "last_id": "claude-opus-5-5"},
        "claude-opus-5-5": {"data": [{"id": "claude-haiku-5-5", "capabilities": {}}],
                            "has_more": False, "last_id": "claude-haiku-5-5"},
    }
    seen = _provider(monkeypatch, lambda r: httpx.Response(200, json=pages[r.url.params.get("after_id")]))
    found = asyncio.run(llm.models("anthropic", "sk-ant-test-key"))
    assert [(m.id, m.label) for m in found] == [("claude-opus-5-5", "Claude Opus 5.5"),
                                                ("claude-haiku-5-5", "claude-haiku-5-5")]
    assert [r.url.params.get("after_id") for r in seen] == [None, "claude-opus-5-5"]


def test_a_model_list_of_an_unknown_shape_is_loud(monkeypatch):
    _provider(monkeypatch, lambda r: httpx.Response(200, json={"data": [{"id": "claude-x"}]}))
    with pytest.raises(llm.ProviderModelsError, match="unexpected model list"):
        asyncio.run(llm.models("anthropic", "sk-ant-test-key"))


def test_a_refusal_names_its_category(monkeypatch):
    _provider(monkeypatch, lambda r: httpx.Response(200, json={
        "model": "claude-opus-5-5", "stop_reason": "refusal",
        "stop_details": {"type": "refusal", "category": "cyber", "explanation": None},
        "content": [], "usage": {"input_tokens": 10, "output_tokens": 0},
    }))
    r = asyncio.run(llm.ping("anthropic", "sk-ant-test-key", "claude-opus-5-5"))
    assert not r.ok
    assert r.error == "stop_reason=refusal (cyber), output_tokens=0"
