from mcp.types import TextContent
from PIL import Image
import base64, io
from fastmcp import FastMCP
from typing import List, Optional
from dotenv import load_dotenv

import sys, os
sys.path.append(os.path.abspath("./agentlego"))

from benchmark import RegionAttributeDescriptionReimplemented 
from agentlego.apis import load_tool 
from agentlego.types import ImageIO, Annotated, Info

import nltk
nltk.data.path.append("./~/nltk_data")

mcp_server = FastMCP("gta-tools")
load_dotenv()

# -----------------------------
# Helpers
# -----------------------------
def decode_image(img_b64: str) -> Image.Image:
    img_bytes = base64.b64decode(img_b64)
    return Image.open(io.BytesIO(img_bytes))

# -----------------------------
# Tool 1: OCR
# -----------------------------
ocr_tool = load_tool("OCR", device="cpu")

@mcp_server.tool(
    name="OCR",
    description=ocr_tool.default_desc
)
async def ocr_tool_wrapper(
    image_path: Annotated[str, Info('The path to the input image.')]
    )  -> list[TextContent]:

    image = ImageIO(image_path)
    result = ocr_tool(image)
    return [TextContent(type="text", text=str(result))]

# -----------------------------
# Tool 2: RegionAttributeDescription
# -----------------------------
region_attribute_description_tool = RegionAttributeDescriptionReimplemented(device="cpu")
region_attribute_description_tool.setup()

@mcp_server.tool(
    name="RegionAttributeDescription",
    description=region_attribute_description_tool.default_desc
)
async def region_attribute_description_tool_wrapper(
        image_path: Annotated[str, Info('The path to the input image.')],
        bbox: Annotated[str, Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')],
        attribute: Annotated[str, Info('The attribute to describe')],
    ) -> list[TextContent]:

    image = ImageIO(image_path)
    result = region_attribute_description_tool(image, bbox, attribute)
    return [TextContent(type="text", text=str(result))]

# -----------------------------
# Tool 3: DetectGivenObject
# -----------------------------
detection_tool = load_tool("ObjectDetection", device="cpu")

@mcp_server.tool(
    name="DetectGivenObject",
    description=detection_tool.default_desc
)
async def detect_given_object_tool_wrapper(   
    image_path: Annotated[str, Info('The path to the input image.')],
    object: Annotated[Optional[str], Info('The object to detect. If not provided, detect all objects.')] = None,
    ) -> list[TextContent]:

    image = ImageIO(image_path)
    result = detection_tool(image)

    # If no object specified, return all results
    if object is None or object.strip() == "":
        return [TextContent(type="text", text=str(result))]

    # Parse and filter results
    lines = str(result).splitlines()
    filtered = [line for line in lines if line.lower().startswith(object.lower())]
    if not filtered:
        return [TextContent(type="text", text=f"No instances of '{object}' found.")]
    return [TextContent(type="text", text="\n".join(filtered))]

# -----------------------------
# Tool 4: ImageDescription
# -----------------------------
image_description_tool = load_tool("ImageDescription", device="cpu")

@mcp_server.tool(
    name="ImageDescription",
    description=image_description_tool.default_desc
)

async def image_description_tool_wrapper(
    image_path: 
        Annotated[str, Info('The path to the input image.')]
    ) -> list[TextContent]:

    image = ImageIO(image_path)
    description = image_description_tool(image)
    return [TextContent(type="text", text=description)]

# -----------------------------
# Tool 5: DrawBox
# -----------------------------
drawbox_tool = load_tool("DrawBox", device="cpu")

@mcp_server.tool(
    name="DrawBox",
    description=drawbox_tool.default_desc
)
async def drawbox_tool_wrapper(
        image_path: Annotated[str, Info('The path to the input image.')],
        bbox: Annotated[str, Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')],
        annotation: Annotated[Optional[str], Info('The extra annotation text of the bbox')] = None,
        ) -> list[TextContent]:
    
    image = ImageIO(image_path)
    result = drawbox_tool(image, bbox, annotation).to_path()
    return [TextContent(type="text", text=str(result))]

# -----------------------------
# Tool 6: AddText
# -----------------------------
addtext_tool = load_tool("AddText", device="cpu")

@mcp_server.tool(
    name="AddText",
    description=addtext_tool.default_desc
)
async def addtext_tool_wrapper(
        image_path: Annotated[str, Info('The path to the input image.')],
        text: Annotated[str, Info('The text to add on the image.')],
        position: Annotated[
            str,
            Info('The left-bottom corner coordinate in the format of `(x, y)`, '
                 'or a combination of ["l"(left), "m"(middle), "r"(right)] '
                 'and ["t"(top), "m"(middle), "b"(bottom)] like "mt" for middle-top')],
        ) -> list[TextContent]:
    
    image = ImageIO(image_path)
    result = addtext_tool(image, text, position).to_path()
    return [TextContent(type="text", text=str(result))]

# -----------------------------
# Tool 7: GoogleSearch
# -----------------------------
search_tool = load_tool("GoogleSearch", device="cpu")

@mcp_server.tool(
    name="GoogleSearch",
    description=search_tool.default_desc
)
async def search_tool_wrapper(
        query: Annotated[str, Info('The search query text.')]
        ) -> list[TextContent]:
    
    result = search_tool(query)
    return [TextContent(type="text", text=result)]

# -----------------------------
# Tool 8: Calculator
# -----------------------------
calculator_tool = load_tool("Calculator", device="cpu")

@mcp_server.tool(
    name="Calculator",
    description=calculator_tool.default_desc
)

async def calculator_tool_wrapper(
        expression: Annotated[str, Info('The mathematical expression to calculate.')]
        ) -> list[TextContent]:
    
    result = calculator_tool(expression)
    return [TextContent(type="text", text=result)]

# -----------------------------
# Tool 9: Plot
# -----------------------------
plot_tool = load_tool("Plot", device="cpu")

@mcp_server.tool(
    name="Plot",
    description=plot_tool.default_desc
)
async def plot_tool_wrapper(
        command: Annotated[str, Info('Markdown format Python code')]
        ) -> list[TextContent]:
    
    result = plot_tool(command).to_path()
    return [TextContent(type="text", text=result)]

# -----------------------------
# Tool 10: MathOCR 
# -----------------------------
mathocr_tool = load_tool("MathOCR", device="cpu")

@mcp_server.tool(
    name="MathOCR",
    description=mathocr_tool.default_desc
)

async def mathocr_tool_wrapper(
        image_path: Annotated[str, Info('Path to the input image.')]
        ) -> list[TextContent]:
    
    image = ImageIO(image_path)
    result = mathocr_tool(image)
    return [TextContent(type="text", text=result)]

# -----------------------------
# Tool 11: CountGivenObject
# -----------------------------
count_tool = load_tool("CountGivenObject", device="cpu")

@mcp_server.tool(
    name="CountGivenObject",
    description=count_tool.default_desc
)
async def count_object_tool_wrapper(
        image_path: Annotated[str, Info('The path to the input image.')],
        text: Annotated[str, Info('The object description in English.')]
        ) -> list[TextContent]:
    
    image = ImageIO(image_path)
    count = count_tool(image, text)
    return [TextContent(type="text", text=str(count))]

# -----------------------------
# Tool 12: Solver
# -----------------------------
solver_tool = load_tool("Solver", device="cpu")

@mcp_server.tool(
    name="Solver",
    description=solver_tool.default_desc
)

async def solver_tool_wrapper(
        command: Annotated[str, Info('Markdown format Python code.')]
        ) -> list[TextContent]:
    
    result = solver_tool(command)
    return [TextContent(type="text", text=result)]

# -----------------------------
# Tool 13: TextToImage
# -----------------------------
text_to_image_tool = load_tool("TextToImage", device="cpu")

@mcp_server.tool(
    name="TextToImage",
    description=text_to_image_tool.default_desc
)

async def text_to_image_tool_wrapper(
        keywords: Annotated[str, Info('A series of English keywords separated by comma.')]
        ) -> list[TextContent]:
    
    result = text_to_image_tool(keywords).to_path()
    return [TextContent(type="text", text=result)]

# -----------------------------
# Tool 14: ImageStylization 
# -----------------------------
image_style_tool = load_tool("ImageStylization", device="cpu")

@mcp_server.tool(
    name="ImageStylization",
    description=image_style_tool.default_desc
)

async def image_style_tool_wrapper(
        image_path: Annotated[str, Info('Path to the input image')],
        instructions: str = Annotated[str, Info('The style instructions in English.')]
        ) -> list[TextContent]:
    
    image = ImageIO(image_path)
    result = image_style_tool(image, instructions).to_path()
    return [TextContent(type="text", text=result)]

# -----------------------------
# Run MCP mcp_server
# -----------------------------
if __name__ == "__main__":
    # Run mcp_server with SSE transport on localhost:8000
    mcp_server.run(transport="sse", host="127.0.0.1", port=8000)
