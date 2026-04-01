import asyncio
import json
import websockets
import time

WS_URL = "ws://localhost:8000/ws"


async def send(ws, payload: dict):
    """Send JSON dict over websocket."""
    await ws.send(json.dumps(payload))


async def recv(ws) -> dict:
    """Receive JSON dict over websocket."""
    return json.loads(await ws.recv())

  
async def main(function):
    async with websockets.connect(WS_URL) as ws:
        # 1) Receive server hello
        # msg = await recv(ws)
        # print("IN:", msg)

        await send(ws, {"type": "session.create", "data": { "name": "TEST" }})
        msg = await recv(ws)
        print("IN:", msg)

        # 2) Ask for sessions list
        await send(ws, {"type": "session.list", "data": {}})
        msg = await recv(ws)
        print("IN:", msg)

        sessions = msg.get("data", {}).get("sessions", [])
        if not sessions:
            print("No sessions exist. Create one from browser first.")
            return

        sid = None
        for session in sessions:
            if session["name"] == "TEST":
                sid = session["session_id"]
                break
        print("Using session: ", sid)

        # 3) Join session as viewer (membership)
        await send(ws, {"type": "session.join", "data": {"session_id": sid}})
        print("IN:", await recv(ws))

        # 4) Request state_controller role
        await send(ws, {"type": "session.role.request", "data": {"session_id": sid, "role": "state_controller"}})
        print("IN:", await recv(ws))

        # 5) Loop: wait for clock.tick, respond with state.push
        while True:
            msg = await recv(ws)
            mtype = msg.get("type")

            if mtype == "clock.tick":
                tick        = msg["data"]
                sim_time    = tick.get("sim_time")
                idx         = tick.get("idx")
                last_update = tick.get("last_update")

                objects = function(sim_time)

                state = {
                    "session_id"    : sid,
                    "sim_time"      : sim_time,
                    "idx"           : idx,
                    "last_update"   : last_update,
                    "latency"       : time.time_ns()*1e-9 - last_update,
                    "objects"       : objects, 
                }

                await send(ws, {"type": "state.push", "data": state})

            elif mtype == "session.list.result":
                data = msg.get("data", {})
                for session in data.get("sessions", []):
                    print(f"Session: {session['session_id']}, name: {session['name']}, viewers: {session['viewer_count']}, clock_master: {session['has_clock_master']}, state_controller: {session['has_state_controller']}")

            else:
                # Print any other server messages
                print("IN:", msg)

def run(function):
    asyncio.run(main(function))
