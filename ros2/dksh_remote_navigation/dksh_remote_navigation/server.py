"""A bounded TCP gateway with one authenticated operator and fresh heartbeats."""
import argparse
import asyncio
import hmac
import ipaddress
import json
import os
import time

from .core import MissionGateway, ProtocolError

MAX_LINE = 2 * 1024 * 1024
MAX_COMMAND = 16384


def decode(line):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError("Duplicate JSON field.")
            result[key] = value
        return result

    def nonfinite(_):
        raise ProtocolError("Non-finite JSON number.")
    try:
        packet = json.loads(line, object_pairs_hook=unique, parse_constant=nonfinite)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise ProtocolError("Invalid JSON packet.") from exc
    if not isinstance(packet, dict):
        raise ProtocolError("Packet must be an object.")
    return packet


async def send(writer, packet):
    line = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if len(line) > MAX_LINE:
        raise ProtocolError("Outgoing packet exceeds the size limit.")
    writer.write(line)
    await asyncio.wait_for(writer.drain(), timeout=1)


class GatewayServer:
    def __init__(self, gateway=None, token=""):
        self.gateway = gateway or MissionGateway()
        self.token = token

    async def handle_connection(self, reader, writer):
        session, sender = "", None
        try:
            hello = decode(await asyncio.wait_for(reader.readline(), timeout=5))
            provided = hello.get("token", "")
            if (type(hello.get("v")) is not int or hello.get("v") != 1 or hello.get("type") != "hello" or
                    not isinstance(provided, str) or len(provided) > 256 or
                    not hmac.compare_digest(provided.encode(), self.token.encode())):
                raise ProtocolError("Authentication or protocol version failed.")
            session = self.gateway.acquire()
            await send(writer, {"v": 1, "type": "welcome", "session": session,
                                "backend": self.gateway.backend.name})
            sender = asyncio.create_task(self.publish(writer, session))
            rate_window, rate_count = time.monotonic(), 0
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=3)
                if not line:
                    break
                if len(line) > MAX_COMMAND:
                    raise ProtocolError("Oversized command packet.")
                now = time.monotonic()
                if now - rate_window >= 1:
                    rate_window, rate_count = now, 0
                rate_count += 1
                if rate_count > 30:
                    raise ProtocolError("Command rate exceeded 30 per second.")
                self.gateway.handle(decode(line))
        except (ProtocolError, ValueError, asyncio.TimeoutError) as exc:
            self.gateway.record("protocol_error", detail=str(exc)[:160])
            try:
                await send(writer, {"v": 1, "type": "error", "detail": str(exc)[:160]})
            except (ConnectionError, asyncio.TimeoutError, ProtocolError):
                pass
        except (ConnectionError, OSError):
            pass
        finally:
            if sender:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            self.gateway.disconnect(session)
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def publish(self, writer, session):
        previous_map = None
        try:
            while self.gateway.session == session:
                current_map = self.gateway.backend.map
                if current_map is not None and current_map is not previous_map:
                    await send(writer, current_map.packet(session))
                    previous_map = current_map
                await send(writer, self.gateway.snapshot())
                await asyncio.sleep(.1)
        except (ConnectionError, OSError, asyncio.TimeoutError, ProtocolError, ValueError):
            self.gateway.disconnect(session)
            writer.close()

    async def tick(self):
        previous = time.monotonic()
        while True:
            now = time.monotonic()
            self.gateway.tick(now - previous)
            previous = now
            await asyncio.sleep(.05)

    async def run(self, host, port):
        server = await asyncio.start_server(self.handle_connection, host, port, limit=MAX_COMMAND)
        tick = asyncio.create_task(self.tick())
        print(f"DKSH simulation gateway ({self.gateway.backend.name}) listening on {host}:{port}", flush=True)
        try:
            async with server:
                serving = asyncio.create_task(server.serve_forever())
                try:
                    done, _ = await asyncio.wait((serving, tick), return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                finally:
                    serving.cancel()
                    await asyncio.gather(serving, return_exceptions=True)
        finally:
            tick.cancel()
            await asyncio.gather(tick, return_exceptions=True)
            self.gateway.stop("Gateway shutting down.")


def main():
    parser = argparse.ArgumentParser(description="Sensor-free DKSH waypoint gateway. Never drives real motors.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backend", choices=("mock", "nav2-sim"), default="mock")
    parser.add_argument("--log-dir", default="logs/remote_navigation")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be 1 to 65535.")
    token = os.environ.get("DKSH_GATEWAY_TOKEN", "")
    try:
        loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = args.host == "localhost"
    if not loopback and len(token) < 16:
        parser.error("Non-loopback binding requires DKSH_GATEWAY_TOKEN (at least 16 characters). Use a private VPN or SSH tunnel.")
    if len(token) > 256:
        parser.error("Token must be at most 256 characters.")
    backend = None
    if args.backend == "nav2-sim":
        from .nav2_backend import Nav2SimulationBackend
        backend = Nav2SimulationBackend()
    from .journal import EventJournal
    journal = EventJournal(args.log_dir)
    service = GatewayServer(MissionGateway(backend, journal=journal), token)
    try:
        asyncio.run(service.run(args.host, args.port))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if backend:
                backend.close()
        finally:
            journal.close()


if __name__ == "__main__":
    main()
