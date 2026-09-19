"""Send a message and receive its acknowledgement over the same TCP connection."""

import argparse
import json
import socket
import uuid
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", help="Laptop IPv4 address, e.g. 192.168.0.15")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--text", help="Send once and exit; otherwise use interactive mode")
    args = parser.parse_args()
    try:
        with socket.create_connection((args.host, args.port), timeout=10) as connection:
            print("Connected. Enter text; /quit closes the connection.")
            with connection.makefile("rb") as incoming:
                while True:
                    value = args.text if args.text is not None else input("Message> ")
                    if value == "/quit":
                        break
                    message = {
                        "type": "telemetry",
                        "message_id": str(uuid.uuid4()),
                        "robot_id": "robot-01",
                        "sent_at": datetime.now(timezone.utc).isoformat(),
                        "data": {"text": value},
                    }
                    payload = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
                    if len(payload) > 65536:
                        raise ValueError("Message exceeds 65536 bytes")
                    connection.sendall(payload)
                    line = incoming.readline(65537)
                    if not line or len(line) > 65536 or not line.endswith(b"\n"):
                        raise ValueError("Connection closed or invalid response")
                    response = json.loads(line.decode("utf-8"))
                    if not isinstance(response, dict):
                        raise ValueError("Invalid response object")
                    if response.get("type") != "ack" or response.get("message_id") != message["message_id"]:
                        raise ValueError(f"Message was not acknowledged: {response}")
                    print("ACK:", json.dumps(response, ensure_ascii=False))
                    if args.text is not None:
                        break
    except (OSError, ValueError) as error:
        parser.exit(1, f"Communication failed: {error}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")


if __name__ == "__main__":
    main()
