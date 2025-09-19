from mcp.types import TextContent
from PIL import Image
import base64, io
from fastmcp import FastMCP

# Import tools from AgentLego
import sys, os
sys.path.append(os.path.abspath("../GTA/agentlego"))

from benchmark import CountGivenObject
from agentlego.tools import OCR, ImageDescription

mcp_server = FastMCP("gta-tools")

# -----------------------------
# Helpers
# -----------------------------
def decode_image(img_b64: str) -> Image.Image:
    img_bytes = base64.b64decode(img_b64)
    return Image.open(io.BytesIO(img_bytes))

# -----------------------------
# Tool 1: OCR
# -----------------------------
ocr_impl = OCR()

@mcp_server.tool(
    name="OCR",
    description="This tool can recognize all text on the input image."
)
async def ocr_tool(params: dict):
    img = decode_image(params["image"])
    # AgentLego OCR usually returns structured text (bbox + content)
    result = ocr_impl(img)
    return [TextContent(type="text", text=str(result))]

# -----------------------------
# Tool 2: CountGivenObject
# -----------------------------
count_impl = CountGivenObject()

@mcp_server.tool(
    name="CountGivenObject",
    description="The tool can count the number of a certain object in the image."
)
async def count_object_tool(params: dict):
    img = decode_image(params["image"])
    obj_name = params["text"]
    count = count_impl(img, obj_name)
    return [TextContent(type="text", text=str(count))]

# -----------------------------
# Tool 3: ImageDescription
# -----------------------------
desc_impl = ImageDescription()

@mcp_server.tool(
    name="ImageDescription",
    description="A useful tool that returns a brief description of the input image."
)
async def describe_tool(params: dict):
    img = decode_image(params["image"])
    description = desc_impl(img)
    return [TextContent(type="text", text=str(description))]

# -----------------------------
# Run MCP mcp_server
# -----------------------------
if __name__ == "__main__":
    # Run mcp_server with SSE transport on localhost:8000
    mcp_server.run(transport="sse", host="127.0.0.1", port=8000)
