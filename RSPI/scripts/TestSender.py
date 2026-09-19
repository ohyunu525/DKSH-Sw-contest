"""라즈베리 파이에서 노트북으로 메시지를 보내고 수신 확인(ACK)을 받습니다."""

import argparse
import json
import socket
import uuid
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "예시: python3 sender.py 192.168.0.15 --text \"테스트\"\n"
            "노트북 IP는 노트북에서 ipconfig를 실행해 Wi-Fi IPv4 주소를 확인합니다."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("host", metavar="노트북_IP", help="노트북의 IPv4 주소. 예: 192.168.0.15")
    parser.add_argument("--port", type=int, default=5000, help="노트북 수신 포트 (기본값: 5000)")
    parser.add_argument("--text", help="한 번만 보낼 메시지. 생략하면 직접 입력 모드")
    args = parser.parse_args()
    try:
        with socket.create_connection((args.host, args.port), timeout=10) as connection:
            print(f"노트북({args.host}:{args.port})에 연결되었습니다.")
            if args.text is None:
                print("보낼 내용을 입력하세요. /quit 입력 시 종료합니다.")
            with connection.makefile("rb") as incoming:
                while True:
                    value = args.text if args.text is not None else input("보낼 메시지> ")
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
                    print("수신 확인(ACK):", json.dumps(response, ensure_ascii=False))
                    if args.text is not None:
                        break
    except (OSError, ValueError) as error:
        parser.exit(1, f"통신 실패: {error}\n")
    except (KeyboardInterrupt, EOFError):
        print("\n송신을 종료했습니다.")


if __name__ == "__main__":
    main()
