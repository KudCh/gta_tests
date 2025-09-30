from mcp.types import TextContent
from PIL import Image
import base64, io
from fastmcp import FastMCP
from typing import List, Optional
from dotenv import load_dotenv

import sys, os
sys.path.append(os.path.abspath("./agentlego"))

from benchmark import RegionAttributeDescriptionReimplemented #, CountGivenObject, ImageDescription
from agentlego.tools import OCR, ObjectDetection, ImageDescription, CountGivenObject, DrawBox, AddText, GoogleSearch, Calculator, Plot, MathOCR, Solver, TextToImage, ImageStylization
from agentlego.utils import load_or_build_object, require
from agentlego.tools import BaseTool
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
ocr_impl = OCR()

@mcp_server.tool(
    name="OCR",
    description=ocr_impl.default_desc
)
async def ocr_tool(
    image_path: Annotated[str, Info('The path to the input image.')]
    )  -> list[TextContent]:

    image = ImageIO(image_path)
    result = ocr_impl(image)
    return [TextContent(type="text", text=str(result))]

# -----------------------------
# Tool 2: RegionAttributeDescription
# -----------------------------
region_attribute_description_impl = RegionAttributeDescriptionReimplemented()

@mcp_server.tool(
    name="RegionAttributeDescription",
    description=region_attribute_description_impl.default_desc
)
async def region_attribute_description_tool(
        image_path: Annotated[str, Info('The path to the input image.')],
        bbox: Annotated[str,
                        Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')],
        attribute: Annotated[str, Info('The attribute to describe')],
    ) -> list[TextContent]:

    image = ImageIO(image_path)
    result = region_attribute_description_impl(image, bbox, attribute)
    return [TextContent(type="text", text=str(result))]

# -----------------------------
# Tool 3: DetectGivenObject
# -----------------------------
detection_impl = ObjectDetection()

@mcp_server.tool(
    name="DetectGivenObject",
    description=detection_impl.default_desc
)
async def detect_given_object_tool(   
    image_path: Annotated[str, Info('The path to the input image.')],
    object: Annotated[Optional[str], Info('The object to detect. If not provided, detect all objects.')] = None,
    ) -> list[TextContent]:

    image = ImageIO(image_path)
    result = detection_impl(image)

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
image_description_impl = ImageDescription()

@mcp_server.tool(
    name="ImageDescription",
    description=image_description_impl.default_desc
)

async def image_description_tool(
    image_path: 
        Annotated[str, Info('The path to the input image.')]
    ) -> list[TextContent]:

    image = ImageIO(image_path)
    description = image_description_impl(image)
    return [TextContent(type="text", text=description)]

# -----------------------------
# Tool 5: DrawBox
# -----------------------------
drawbox_impl = DrawBox()

@mcp_server.tool(
    name="DrawBox",
    description=drawbox_impl.default_desc
)
async def drawbox_tool(
        image_path: Annotated[str, Info('The path to the input image.')],
        bbox: Annotated[str, Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')],
        annotation: Annotated[Optional[str], Info('The extra annotation text of the bbox')] = None,
        ) -> list[TextContent]:
    
    image = ImageIO(image_path)
    result = drawbox_impl(image, bbox, annotation).to_path()
    return [TextContent(type="text", text=str(result))]

# -----------------------------
# Tool 6: AddText
# -----------------------------
addtext_impl = AddText()

@mcp_server.tool(
    name="AddText",
    description=addtext_impl.default_desc
)
async def addtext_tool(
        image_path: Annotated[str, Info('The path to the input image.')],
        text: Annotated[str, Info('The text to add on the image.')],
        position: Annotated[
            str,
            Info('The left-bottom corner coordinate in the format of `(x, y)`, '
                 'or a combination of ["l"(left), "m"(middle), "r"(right)] '
                 'and ["t"(top), "m"(middle), "b"(bottom)] like "mt" for middle-top')],
        ) -> list[TextContent]:
    
    image = ImageIO(image_path)
    result = addtext_impl(image, text, position).to_path()
    return [TextContent(type="text", text=str(result))]

# -----------------------------
# Tool 7: GoogleSearch
# -----------------------------
search_impl = GoogleSearch()

@mcp_server.tool(
    name="GoogleSearch",
    description=search_impl.default_desc
)
async def search_tool(
        query: Annotated[str, Info('The search query text.')]
        ) -> list[TextContent]:
    
    result = search_impl(query)
    return [TextContent(type="text", text=result)]

# -----------------------------
# Tool 8: Calculator
# -----------------------------
calculator_impl = Calculator()

@mcp_server.tool(
    name="Calculator",
    description=calculator_impl.default_desc
)

async def calculator_tool(
        expression: Annotated[str, Info('The mathematical expression to calculate.')]
        ) -> list[TextContent]:
    
    result = calculator_impl(expression)
    return [TextContent(type="text", text=result)]

# -----------------------------
# Tool 9: Plot
# -----------------------------
plot_impl = Plot()

@mcp_server.tool(
    name="Plot",
    description=plot_impl.default_desc
)
async def plot_tool(
        command: Annotated[str, Info('Markdown format Python code')]
        ) -> list[TextContent]:
    
    result = plot_impl(command).to_path()
    return [TextContent(type="text", text=result)]

# -----------------------------
# Tool 10: MathOCR 
# -----------------------------
mathocr_impl = MathOCR()

@mcp_server.tool(
    name="MathOCR",
    description=mathocr_impl.default_desc
)

async def mathocr_tool(
        image_path: Annotated[str, Info('Path to the input image.')]
        ) -> list[TextContent]:
    
    image = ImageIO(image_path)
    result = mathocr_impl(image)
    return [TextContent(type="text", text=result)]

# -----------------------------
# Tool 11: CountGivenObject
# -----------------------------
count_impl = CountGivenObject()

@mcp_server.tool(
    name="CountGivenObject",
    description=count_impl.default_desc
)
async def count_object_tool(
        image_path: Annotated[str, Info('The path to the input image.')],
        text: Annotated[str, Info('The object description in English.')]
        ) -> list[TextContent]:
    
    image = ImageIO(image_path)
    count = count_impl(image, text)
    return [TextContent(type="text", text=str(count))]

# -----------------------------
# Tool 12: Solver
# -----------------------------
solver_impl = Solver()

@mcp_server.tool(
    name="Solver",
    description=solver_impl.default_desc
)

async def solver_tool(
        command: Annotated[str, Info('Markdown format Python code.')]
        ) -> list[TextContent]:
    
    result = solver_impl(command)
    return [TextContent(type="text", text=result)]

# -----------------------------
# Tool 13: TextToImage
# -----------------------------
text_to_image_impl = TextToImage()

@mcp_server.tool(
    name="TextToImage",
    description=text_to_image_impl.default_desc
)

async def text_to_image_tool(
        keywords: Annotated[str, Info('A series of English keywords separated by comma.')]
        ) -> list[TextContent]:
    
    result = text_to_image_impl(keywords).to_path()
    return [TextContent(type="text", text=result)]

# -----------------------------
# Tool 14: ImageStylization 
# -----------------------------
image_style_impl = ImageStylization()

@mcp_server.tool(
    name="ImageStylization",
    description=image_style_impl.default_desc
)

async def image_style_tool(
        image_path: Annotated[str, Info('Path to the input image')],
        instructions: str
        ) -> list[TextContent]:
    
    image = ImageIO(image_path)
    result = image_style_impl(image, instructions).to_path()
    return [TextContent(type="text", text=result)]

# -----------------------------
# Run MCP mcp_server
# -----------------------------
if __name__ == "__main__":
    # Run mcp_server with SSE transport on localhost:8000
    mcp_server.run(transport="sse", host="127.0.0.1", port=8000)
