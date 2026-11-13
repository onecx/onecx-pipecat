import json

import httpx
import pytest

from onecx_pipecat.adapters import HttpChatPersistence, InMemoryPersistence
from onecx_pipecat.adapters.persistence import PersistenceError
from onecx_pipecat.config import ChatSettings
from onecx_pipecat.models import VoiceEntry, VoiceEntryKind, VoiceEntryStatus


def _persistence(handler, service_auth: httpx.Auth | None = None) -> HttpChatPersistence:
    adapter = HttpChatPersistence(ChatSettings(base_url="http://chat"), service_auth)
    adapter.set_client(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return adapter


async def test_saves_a_human_entry_as_a_user_message():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["token"] = request.headers["apm-principal-token"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, headers={"Location": "http://chat/x/messages/m-9"})

    entry_id = await _persistence(handler).append(
        "c 1",
        VoiceEntry(kind=VoiceEntryKind.HUMAN, text="hello", agent_config_version="config-v2"),
        apm_principal_token="tok",
        user_id="user-1",
    )
    assert entry_id == "m-9"
    assert seen["url"] == "http://chat/internal/chats/c%201/messages"
    assert seen["method"] == "POST"
    assert seen["token"] == "tok"
    assert seen["body"] == {
        "type": "HUMAN",
        "text": "hello",
        "userId": "user-1",
        "skipAIProcessing": True,
        "agentConfigVersion": "config-v2",
    }


async def test_saves_an_interrupted_assistant_entry_as_the_spoken_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "m-2", "type": "ASSISTANT"})

    entry = VoiceEntry(
        kind=VoiceEntryKind.ASSISTANT,
        text="Sentence 1.",
        status=VoiceEntryStatus.INTERRUPTED,
        agent_config_version="config-v1",
    )
    assert await _persistence(handler).append("c", entry) == "m-2"
    assert seen["body"] == {
        "type": "ASSISTANT",
        "text": "Sentence 1.",
        "skipAIProcessing": True,
        "agentConfigVersion": "config-v1",
    }


async def test_requests_carry_the_service_auth():
    class Bearer(httpx.Auth):
        def auth_flow(self, request):
            request.headers["Authorization"] = "Bearer svc"
            yield request

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return (
            httpx.Response(201, json={"id": "m"})
            if request.method == "POST"
            else httpx.Response(200, json=[])
        )

    adapter = _persistence(handler, Bearer())
    await adapter.append("c", VoiceEntry(kind=VoiceEntryKind.ASSISTANT, text="x"))
    await adapter.history("c")
    assert seen == ["Bearer svc", "Bearer svc"]


@pytest.mark.parametrize(
    "response", [httpx.Response(404), httpx.Response(500), httpx.Response(201)]
)
async def test_save_failures_raise(response):
    with pytest.raises(PersistenceError):
        await _persistence(lambda request: response).append(
            "c", VoiceEntry(kind=VoiceEntryKind.HUMAN, text="x"), user_id="u"
        )


async def test_save_transport_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(PersistenceError):
        await _persistence(handler).append("c", VoiceEntry(kind=VoiceEntryKind.HUMAN, text="x"))


async def test_non_chat_entry_kinds_are_rejected():
    adapter = _persistence(lambda request: httpx.Response(201, json={"id": "m"}))
    with pytest.raises(PersistenceError):
        await adapter.append("c", VoiceEntry(kind=VoiceEntryKind.TOOL_RESULT, text="x"))


async def test_history_orders_filters_and_truncates():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["token"] = request.headers["apm-principal-token"]
        return httpx.Response(
            200,
            json=[
                {"type": "ASSISTANT", "text": "third", "creationDate": "2026-01-01T10:00:03Z"},
                {"type": "SYSTEM", "text": "hidden", "creationDate": "2026-01-01T10:00:00Z"},
                {"type": "HUMAN", "text": "first", "creationDate": "2026-01-01T10:00:01Z"},
                {"type": "TOOL", "text": "unknown", "creationDate": "2026-01-01T10:00:02Z"},
                {"type": "HUMAN", "text": "  ", "creationDate": "2026-01-01T10:00:02Z"},
                {"type": "HUMAN", "text": "second", "creationDate": "2026-01-01T10:00:02+00:00"},
            ],
        )

    adapter = _persistence(handler)
    assert await adapter.history("c", apm_principal_token="tok", limit=50) == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "third"},
    ]
    assert seen == {"url": "http://chat/internal/chats/c/messages", "token": "tok"}
    assert await adapter.history("c", limit=2) == [
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "third"},
    ]


async def test_history_of_a_chat_without_messages_is_empty():
    assert await _persistence(lambda request: httpx.Response(404)).history("c") == []


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(403),
        httpx.Response(500),
        httpx.Response(200, text="nope"),
        httpx.Response(200, json={}),
    ],
)
async def test_history_failures_raise(response):
    with pytest.raises(PersistenceError):
        await _persistence(lambda request: response).history("c")


async def test_in_memory_history_maps_visible_human_and_assistant_entries():
    store = InMemoryPersistence()
    await store.append("c", VoiceEntry(kind=VoiceEntryKind.HUMAN, text="hi"))
    await store.append("c", VoiceEntry(kind=VoiceEntryKind.TOOL_RESULT, text="skip"))
    await store.append("c", VoiceEntry(kind=VoiceEntryKind.ASSISTANT, text="yo"))
    assert await store.history("c") == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
