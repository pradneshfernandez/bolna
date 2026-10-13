"""Rime streaming: one end-of-stream per reply, after its last audio.

Frames follow what Rime's /ws2 endpoint sends with segment=never: for each non-empty flush, the audio as
"chunk" events, then "timestamps" and "done", all tagged with the contextId of the text that produced them.
sender() moves to a new contextId once a reply's text is all sent, so every frame of that reply carries the
old one. The receiver used to end the reply on any frame with an old contextId, i.e. on its first frame."""

import asyncio
import base64
import json
from unittest.mock import MagicMock

import websockets
import websockets.protocol

from bolna.synthesizer.rime_synthesizer import RimeSynthesizer


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.inbox = asyncio.Queue()
        self.state = websockets.protocol.State.OPEN

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def recv(self):
        return await self.inbox.get()

    def frames(self, ctx, payloads):
        """What Rime sends for one non-empty flush."""
        for payload in payloads:
            self.inbox.put_nowait(
                json.dumps({"type": "chunk", "data": base64.b64encode(payload).decode(), "contextId": ctx})
            )
        self.inbox.put_nowait(json.dumps({"type": "timestamps", "word_timestamps": {}, "contextId": ctx}))
        self.inbox.put_nowait(json.dumps({"type": "done", "done": True, "contextId": ctx}))


def make_synth():
    synth = RimeSynthesizer(
        voice_id="cove", voice="cove", model="mistv2", caching=False, stream=True, synthesizer_key="k"
    )
    synth.task_manager_instance = MagicMock()
    synth.task_manager_instance.is_sequence_id_in_current_ids.return_value = True
    synth._process_audio_chunk = lambda chunk: chunk
    synth.websocket = FakeSocket()
    return synth


async def push(synth, text, seq, end=False):
    await synth.push({"data": text, "meta_info": {"sequence_id": seq, "turn_id": seq, "end_of_llm_stream": end}})
    await synth.sender_task


async def collect(synth, until_eos=1, timeout=1.0):
    out = []
    gen = synth.generate()
    try:
        while sum(1 for data, _ in out if data == b"\x00") < until_eos:
            pkt = await asyncio.wait_for(gen.__anext__(), timeout)
            out.append((pkt["data"], pkt["meta_info"].get("sequence_id")))
    except asyncio.TimeoutError:
        pass
    finally:
        await gen.aclose()
    return out


def reply_context(synth):
    return next(m["contextId"] for m in synth.websocket.sent if "text" in m)


def test_asks_rime_for_one_done_per_flush():
    assert "segment=never" in make_synth().ws_url


async def test_reply_ends_once_after_its_last_audio():
    synth = make_synth()
    await push(synth, "Sure, I can book that.", 10)
    await push(synth, " Shall I confirm it for 8 pm?", 10, end=True)
    ctx = reply_context(synth)
    synth.websocket.frames(ctx, [b"A1", b"A2"])  # first flush's "done" arrives after the reply's end was sent
    synth.websocket.frames(ctx, [b"B1", b"B2"])

    out = await collect(synth, until_eos=2, timeout=0.3)

    assert out == [(b"A1", 10), (b"A2", 10), (b"B1", 10), (b"B2", 10), (b"\x00", 10)]


async def test_frames_tagged_with_the_old_context_are_played_not_ended():
    synth = make_synth()
    await push(synth, "Hello there.", 10, end=True)
    ctx = reply_context(synth)
    assert synth.context_id != ctx  # sender already moved on
    synth.websocket.frames(ctx, [b"A1", b"A2", b"A3"])

    out = await collect(synth)

    assert [data for data, _ in out] == [b"A1", b"A2", b"A3", b"\x00"]


async def test_text_less_end_after_the_audio_finished_still_ends_the_reply():
    """Rime sends no "done" for an empty flush, so nothing would arrive to end the reply."""
    synth = make_synth()
    await push(synth, "Is there anything else?", 10)
    synth.websocket.frames(reply_context(synth), [b"A1"])
    gen = synth.generate()
    assert (await gen.__anext__())["data"] == b"A1"

    await push(synth, "", 10, end=True)
    eos = await asyncio.wait_for(gen.__anext__(), 1)

    assert eos["data"] == b"\x00"
    assert eos["meta_info"]["end_of_synthesizer_stream"] is True
    await gen.aclose()


async def test_interrupted_reply_does_not_end_the_next_one():
    synth = make_synth()
    await push(synth, "Sure, let me check.", 10, end=True)
    old_ctx = reply_context(synth)
    await synth.handle_interruption()
    synth.websocket.frames(old_ctx, [b"X1"])  # in-flight audio and "done" of the interrupted reply

    out = await collect(synth, timeout=0.3)

    assert b"\x00" not in [data for data, _ in out]
    assert synth.current_turn_start_time is None


async def test_socket_closed_by_rime_mid_reply_ends_the_reply():
    """Rime closes the socket (1011) on text it can't speak, e.g. an emoji-only push; no "done" follows."""
    from websockets.exceptions import ConnectionClosedError

    synth = make_synth()
    await push(synth, "Great!", 10)
    await push(synth, " 😀", 10, end=True)
    synth.websocket.inbox.put_nowait(
        json.dumps({"type": "chunk", "data": base64.b64encode(b"A1").decode(), "contextId": reply_context(synth)})
    )

    async def closed():
        raise ConnectionClosedError(None, None)

    gen = synth.generate()
    assert (await gen.__anext__())["data"] == b"A1"
    synth.websocket.recv = closed
    eos = await asyncio.wait_for(gen.__anext__(), 1)

    assert eos["data"] == b"\x00"
    assert eos["meta_info"]["end_of_synthesizer_stream"] is True
    assert not synth.pending_dones and not synth.finished_contexts
    await gen.aclose()


async def test_idle_close_emits_no_end_of_stream():
    from websockets.exceptions import ConnectionClosedError

    synth = make_synth()

    async def closed():
        raise ConnectionClosedError(None, None)

    synth.websocket.recv = closed
    assert [p async for p in synth.generate()] == []
