# mcp_openapi_bridge.py
import asyncio
import ast
import json
import re
import uuid
from typing import Dict, Any, Optional

import httpx
from fastapi import FastAPI, Body, HTTPException
from pydantic import RootModel

MCP_SSE_URL = "http://127.0.0.1:8000/sse"  # change if different
BRIDGE_HOST = "http://127.0.0.1:9000"      # where this bridge will run
POST_BASE_URL: Optional[str] = None        # discovered from SSE 'endpoint' event
TOOLS: Dict[str, dict] = {}                # tool_name -> {description, inputSchema}
_pending_calls: Dict[str, asyncio.Future] = {}  # call_id -> Future

app = FastAPI(title="FastMCP -> OpenAPI Bridge")


class GenericPayload(RootModel):
    pass


# ---------- Helpers for parsing SSE messages ----------
def _find_balanced_braces(s: str, start_idx: int) -> Optional[int]:
    """Find index of closing brace matching the opening brace at start_idx."""
    if start_idx >= len(s) or s[start_idx] != "{":
        return None
    depth = 0
    for i in range(start_idx, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def parse_tool_list_from_text(text: str):
    """
    Heuristic parser to find Tool(...) entries in SSE text and extract name + inputSchema.
    The server prints Python repr like:
      Tool(name='OCR', title=None, description='...', inputSchema={'properties': ...}, outputSchema=None, ...)
    We'll extract `name` and `inputSchema`.
    """
    for m in re.finditer(r"Tool\(", text):
        start = m.start()
        # find "name='...'"
        name_match = re.search(r"name\s*=\s*'([^']+)'", text[start:start+500])
        if not name_match:
            continue
        tool_name = name_match.group(1)

        # find "inputSchema="
        input_schema_index = text.find("inputSchema=", start)
        if input_schema_index == -1:
            continue
        brace_index = text.find("{", input_schema_index)
        if brace_index == -1:
            continue
        end_brace = _find_balanced_braces(text, brace_index)
        if end_brace is None:
            continue
        schema_str = text[brace_index:end_brace+1]
        try:
            # The repr uses single quotes and Python atoms; ast.literal_eval converts to python dict
            schema_obj = ast.literal_eval(schema_str)
        except Exception as e:
            # fallback: try lightweight JSON-ish fix
            try:
                schema_json = schema_str.replace("'", '"')
                schema_obj = json.loads(schema_json)
            except Exception:
                # can't parse, skip
                continue

        # store tool definition
        TOOLS[tool_name] = {
            "description": (re.search(r"description\s*=\s*'([^']*)'", text[start:start+800]) or {}).group(1) if re.search(r"description\s*=\s*'([^']*)'", text[start:start+800]) else "",
            "inputSchema": schema_obj,
        }


def parse_call_result_from_text(text: str):
    """
    Heuristically detect call results. For lines containing `CallToolResult(...)` produce
    a dictionary with fields. This is a pragmatic parser for the sample output.
    """
    # Try to catch call results like: "here is the response for call_tool("OCR", {"image_path": "gta_dataset\\image\\image_1.jpg"}): CallToolResult(content=[TextContent(... text='...')])"
    # We'll attempt to find the tool name and text payload.
    m = re.search(r"CallToolResult\((.*?)\)", text, re.S)
    if not m:
        return None
    payload_text = m.group(1)
    # try to extract textual content inside TextContent(... text='...') or similar
    text_match = re.search(r"text\s*=\s*'([^']*)'", payload_text)
    if text_match:
        return {"raw": text, "text": text_match.group(1)}
    # fallback: return raw payload
    return {"raw": text, "text": payload_text}


# ---------- SSE reader task (background) ----------
async def sse_listener_loop():
    """
    Connect to the MCP SSE stream and parse messages:
      - discover POST endpoint from 'endpoint' event
      - parse tool list and update TOOLS
      - parse tool call results and resolve pending futures
    """
    global POST_BASE_URL
    print(f"[bridge] Connecting to MCP SSE at {MCP_SSE_URL} ...")
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream("GET", MCP_SSE_URL) as resp:
                if resp.status_code != 200:
                    print(f"[bridge] SSE connection failed: {resp.status_code}")
                    return
                # stream raw lines
                event = None
                data_lines = []
                async for raw_line in resp.aiter_lines():
                    if raw_line is None:
                        continue
                    line = raw_line.strip()
                    if not line:
                        # blank line indicates dispatch of event
                        if event and data_lines:
                            data = "\n".join(data_lines)
                            # handle event types
                            if event == "endpoint":
                                # data example: /messages/?session_id=ebda...
                                ep = data.strip()
                                # make absolute URL
                                if ep.startswith("http"):
                                    POST_BASE_URL = ep
                                else:
                                    # compose from MCP_SSE_URL base
                                    base = MCP_SSE_URL.split("/sse")[0]
                                    POST_BASE_URL = base.rstrip("/") + "/" + ep.lstrip("/")
                                print(f"[bridge] Discovered post endpoint: {POST_BASE_URL}")
                            else:
                                # scan for tool list in data
                                if "response for list_tools" in data or "Tool(" in data:
                                    parse_tool_list_from_text(data)
                                    if TOOLS:
                                        print(f"[bridge] Discovered tools: {list(TOOLS.keys())}")
                                # scan for call results
                                cr = parse_call_result_from_text(data)
                                if cr:
                                    # try to resolve pending futures: naive match by tool name existence in text
                                    for call_id, fut in list(_pending_calls.items()):
                                        if not fut.done():
                                            # if the call id appears in text, use it; else fallback use textual heuristics
                                            if call_id in data or any(k in data for k in TOOLS.keys()):
                                                fut.set_result(cr)
                                                _pending_calls.pop(call_id, None)
                                                break
                        event = None
                        data_lines = []
                        continue

                    if line.startswith("event:"):
                        event = line.split("event:")[1].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line.split("data:", 1)[1].strip())
                    else:
                        # other SSE fields ignored
                        continue
        except Exception as e:
            print("[bridge] SSE listener error:", e)
            # restart after a delay
            await asyncio.sleep(1)
            asyncio.create_task(sse_listener_loop())


@app.on_event("startup")
async def startup_event():
    # start background SSE listener
    asyncio.create_task(sse_listener_loop())


# ---------- OpenAPI & tool endpoints ----------
@app.get("/openapi.json")
async def openapi_spec():
    """
    Generate a minimal OpenAPI spec from discovered TOOLS.
    """
    paths = {}
    for tool_name, info in TOOLS.items():
        schema = info.get("inputSchema") or {"type": "object", "properties": {}}
        paths[f"/tools/{tool_name}"] = {
            "post": {
                "summary": info.get("description", ""),
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": schema
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Tool response", "content": {"application/json": {}}}
                }
            }
        }
    return {
        "openapi": "3.0.0",
        "info": {"title": "MCP Bridge", "version": "1.0"},
        "paths": paths,
    }


@app.post("/tools/{tool_name}")
async def call_tool_endpoint(tool_name: str, payload: GenericPayload = Body(...)):
    """
    Called by GTA/OpenCompass. Forwards to MCP: sends a call_tool event and awaits SSE response.
    """
    if tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")

    # Ensure we have a POST endpoint to send MCP events
    if not POST_BASE_URL:
        raise HTTPException(status_code=503, detail="MCP POST endpoint not discovered yet via SSE")

    call_id = str(uuid.uuid4())
    event_payload = {
        "id": call_id,
        "type": "call_tool",
        "tool": tool_name,
        "arguments": payload.root,
    }

    # create a future and register
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    _pending_calls[call_id] = fut

    # POST the event to the messages endpoint
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # POST as JSON - FastMCP typically accepts JSON event posts to the messages endpoint
            r = await client.post(POST_BASE_URL, json=event_payload)
        except Exception as e:
            _pending_calls.pop(call_id, None)
            raise HTTPException(status_code=500, detail=f"Failed to send event to MCP: {e}")

    # simple wait for response
    try:
        result = await asyncio.wait_for(fut, timeout=25.0)
    except asyncio.TimeoutError:
        _pending_calls.pop(call_id, None)
        raise HTTPException(status_code=504, detail="Timed out waiting for MCP tool result")

    # result is a dict like {"text": "..."} or {"raw": "..."}
    return {"call_id": call_id, "result": result}


@app.get("/tools")
async def list_tools():
    return {"tools": list(TOOLS.keys())}
