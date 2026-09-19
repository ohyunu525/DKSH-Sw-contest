"""Receive newline-delimited JSON over TCP. Python 3, standard library only."""

import argparse
import json
import socketserver
from datetime import datetime, timezone

MAX_MESSAGE_BYTES = 65536


class Receiver(socketserver.StreamRequestHandler):
    def handle(self):
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        print(f"CONNECTED {peer}", flush=True)
        self.request.settimeout(120)
        try:
            while True:
                # TCP is a byte stream; a newline marks each complete message.
                line = self.rfile.readline(MAX_MESSAGE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
                    self.reply({"type": "error", "error": "message_too_large_or_incomplete"})
                    break
                try:
                    message = json.loads(line.decode("utf-8"))
                    if not isinstance(message, dict):
                        raise ValueError("Expected a JSON object")
                    if not isinstance(message.get("message_id"), str):
                        raise ValueError("message_id must be a string")
                    if message.get("type") != "telemetry":
                        raise ValueError("type must be telemetry")
                except (ValueError, UnicodeError) as error:
                    self.reply({"type": "error", "error": str(error)})
                    continue

                print(f"RECEIVED {peer}: {json.dumps(message, ensure_ascii=True)}", flush=True)
                self.reply({
                    "type": "ack",
                    "message_id": message["message_id"],
                    "status": "received",
                    "received_at": datetime.now(timezone.utc).isoformat(),
                })
        except OSError as error:
            print(f"CONNECTION ENDED {peer}: {error}", flush=True)
        finally:
            print(f"DISCONNECTED {peer}", flush=True)

    def reply(self, message):
        self.wfile.write((json.dumps(message) + "\n").encode("utf-8"))
        self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    try:
        with Server((args.host, args.port), Receiver) as server:
            print(f"Listening on {args.host}:{args.port}; Ctrl+C to stop.", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    except OSError as error:
        parser.exit(1, f"Cannot start receiver: {error}\n")


if __name__ == "__main__":
    main()
