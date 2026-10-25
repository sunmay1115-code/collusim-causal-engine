"""Small OpenAI-compatible server for deterministic starter-code validation."""

from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _agent_index(text: str, role: str) -> int:
    match = re.search(rf"Your name is {role} ([^\.]+)\.", text)
    name = match.group(1) if match else role
    return sum(ord(character) for character in name) % 3


def _pricing_response(text: str) -> dict[str, Any]:
    visible_rounds = len(re.findall(r"^Round \d+$", text, flags=re.MULTILINE))
    if visible_rounds == 0:
        price = 1.45
    elif visible_rounds >= 5:
        price = 1.82
    else:
        price = 1.54
    return {
        "observations": f"Visible market rounds: {visible_rounds}.",
        "plans": "Maintain a stable supracompetitive price when history is rich.",
        "insights": "More visible history supports price coordination.",
        "chosen_price": price,
    }


def _auction_response(text: str, buyer: bool) -> dict[str, Any]:
    role = "Buyer" if buyer else "Seller"
    index = _agent_index(text, role)
    hour_match = re.search(r"This is Hour #(\d+)", text)
    hour = int(hour_match.group(1)) if hour_match else 1
    if buyer:
        bid = 95.0 + index
        return {
            "reflection": "Use a stable bid to keep execution predictable.",
            "plan_for_this_hour": "Submit a competitive buyer bid.",
            "bid": bid,
            "new_memory": f"Hour {hour}: bid {bid:.2f}.",
            "scratch_pad_update": "#### Observations\nStable demand.",
        }

    message_match = re.search(
        r"maximum length limit of (\d+) characters",
        text,
    )
    cap = int(message_match.group(1)) if message_match else 500
    if cap <= 10:
        ask = [88.0, 99.0, 94.0][index]
        message = f"hold{int(ask)}"
    else:
        ask = 90.0 + 0.5 * index
        message = f"Sellers should hold near {ask:.2f} and avoid undercutting."
    return {
        "reflection": "Keep the ask above marginal cost.",
        "plan_for_this_hour": "Preserve margin while seeking execution.",
        "ask": ask,
        "new_memory": f"Hour {hour}: ask {ask:.2f}.",
        "plan_for_message": "Signal a stable price.",
        "message_to_sellers": message,
        "scratch_pad_update": "#### Observations\nTrack queue depth.",
    }


def response_content(messages: list[dict[str, Any]]) -> str:
    text = "\n".join(str(message.get("content", "")) for message in messages)
    if '"chosen_price"' in text:
        payload = _pricing_response(text)
    elif "Your name is Buyer " in text:
        payload = _auction_response(text, buyer=True)
    elif "Your name is Seller " in text:
        payload = _auction_response(text, buyer=False)
    else:
        payload = {
            "reflection": "No action.",
            "plan_for_this_hour": "No action.",
            "ask": None,
            "bid": None,
            "new_memory": "",
            "scratch_pad_update": "",
        }
    return json.dumps(payload)


class MockHandler(BaseHTTPRequestHandler):
    server_version = "StarterMock/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"/v1/models", "/models"}:
            payload = {
                "object": "list",
                "data": [
                    {
                        "id": "starter-mock",
                        "object": "model",
                        "created": 0,
                        "owned_by": "local",
                    }
                ],
            }
            self._send_json(payload)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in {
            "/v1/chat/completions",
            "/chat/completions",
        }:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        content = response_content(request.get("messages", []))
        payload = {
            "id": f"chatcmpl-mock-{id(request)}",
            "object": "chat.completion",
            "created": 0,
            "model": request.get("model", "starter-mock"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        self._send_json(payload)

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def make_server(port: int = 0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), MockHandler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = make_server(args.port)
    print(f"serving on http://127.0.0.1:{server.server_port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
