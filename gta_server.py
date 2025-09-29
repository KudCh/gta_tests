# -----------------------------
# Monkey patch for transformers.AutoModel.from_pretrained
import transformers

old_from_pretrained = transformers.AutoModel.from_pretrained

def patched_from_pretrained(*args, **kwargs):
    kwargs.setdefault("attn_implementation", "eager")
    return old_from_pretrained(*args, **kwargs)

transformers.AutoModel.from_pretrained = patched_from_pretrained

old_causal_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
def patched_causal_from_pretrained(*args, **kwargs):
    kwargs.setdefault("attn_implementation", "eager")
    return old_causal_from_pretrained(*args, **kwargs)

transformers.AutoModelForCausalLM.from_pretrained = patched_causal_from_pretrained
# -----------------------------

from mcp.types import TextContent
from PIL import Image
import base64, io
from fastmcp import FastMCP
from typing import List, Optional
from dotenv import load_dotenv

# Import tools from AgentLego
import sys, os
sys.path.append(os.path.abspath("./agentlego"))

from benchmark import RegionAttributeDescriptionReimplemented #, CountGivenObject, ImageDescription
from agentlego.tools import OCR, ObjectDetection, ImageDescription, CountGivenObject, DrawBox, AddText, GoogleSearch, Calculator, Plot, MathOCR, Solver, TextToImage, ImageStylization
from agentlego.utils import load_or_build_object, require
from agentlego.tools import BaseTool
from agentlego.types import ImageIO, Annotated, Info


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

"""
OCR tool implementation

def apply(
        self,
        image: ImageIO,
    ) -> Annotated[str,
                   Info('OCR results, include bbox in x1, y1, x2, y2 format '
                        'and the recognized text.')]:
"""

@mcp_server.tool(
    name="OCR",
    description="This tool can recognize all text on the input image."
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

"""
RegionAttributeDescription tool implementation

    def apply(
        self,
        image: ImageIO,
        bbox: Annotated[str,
                        Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')],
        attribute: Annotated[str, Info('The attribute to describe')],
    ) -> str:
"""

@mcp_server.tool(
    name="RegionAttributeDescription",
    description="Describe the attribute of a region of the input image."
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
"""
    def apply(
        self,
        image: ImageIO,
    ) -> Annotated[str,
                   Info('All detected objects, include object name, '
                        'bbox in (x1, y1, x2, y2) format, '
                        'and detection score.')]:
"""

@mcp_server.tool(
    name="DetectGivenObject",
    description="This tool can detect all instances of a given object on the input image. The tool can only detect objects defined in COCO 80 classes"
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

"""
    def apply(self, image: ImageIO) -> str:
"""

@mcp_server.tool(
    name="ImageDescription",
    description="A useful tool that returns a brief description of the input image."
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

"""
    def apply(
        self,
        image: ImageIO,
        bbox: Annotated[str,
                        Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')],
        annotation: Annotated[Optional[str],
                              Info('The extra annotation text of the bbox')] = None,
    ) -> ImageIO:
"""
@mcp_server.tool(
    name="DrawBox",
    description="A tool to draw a box on a certain region of the input image."
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

"""
    def apply(
        self,
        image: ImageIO,
        text: str,
        position: Annotated[
            str,
            Info('The left-bottom corner coordinate in the format of `(x, y)`, '
                 'or a combination of ["l"(left), "m"(middle), "r"(right)] '
                 'and ["t"(top), "m"(middle), "b"(bottom)] like "mt" for middle-top')],
        color: str = 'red',
    ) -> ImageIO:

"""
@mcp_server.tool(
    name="AddText",
    description="A tool to add text to the input image."
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

"""
    def apply(self, query: str) -> str:

"""
@mcp_server.tool(
    name="GoogleSearch",
    description='The tool can search the input query text from Google and return the related results.'
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

"""
def apply(self, expression: str) -> str:
"""
@mcp_server.tool(
    name="Calculator",
    description='A calculator tool. The input must be a single Python '
                    'expression and you cannot import packages. You can use functions '
                    'in the `math` package without import.'
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

"""
    def apply(self, command: Annotated[str,
                                       Info('Markdown format Python code')]) -> ImageIO:
"""
@mcp_server.tool(
    name="Plot",
    description='''\
This tool can execute Python code to plot diagrams. The code should include a function named 'solution'. The function should return the matplotlib figure directly. Avoid printing the answer. The code instance format is as follows:

```python
# import packages
import matplotlib.pyplot as plt
def solution():
    # labels and data
    cars = ['AUDI', 'BMW', 'FORD', 'TESLA', 'JAGUAR', 'MERCEDES']
    data = [23, 17, 35, 29, 12, 41]

    # draw diagrams
    figure = plt.figure(figsize=(8, 6))
    plt.pie(data, labels=cars, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Car Distribution')
    return figure
```
'''  
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

"""
 def apply(self, image: ImageIO) -> str:
"""
@mcp_server.tool(
    name="MathOCR",
    description='This tool can recognize math expressions from an '
                    'image and return the latex style expression.' 
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

"""
    def apply(
        self,
        image: ImageIO,
        text: Annotated[str, Info('The object description in English.')],
        bbox: Annotated[Optional[str],
                        Info('The bbox coordinate in the format of `(x1, y1, x2, y2)`')] = None,
    ) -> int:
"""
@mcp_server.tool(
    name="CountGivenObject",
    description="The tool can count the number of a certain object in the image."
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

"""
    def apply(self, command: Annotated[str, Info('Markdown format Python code')]) -> str:
"""
@mcp_server.tool(
    name="Solver",
    description='''\
This tool can execute Python code to solve math equations. The code should include a function named 'solution'. You should use the `sympy` library in your code to solve the equations. The function should return its answer in str format. Avoid printing the answer. The code instance format is as follows:

```python
# import packages
from sympy import symbols, Eq, solve
def solution():
    # Define symbols
    x, y = symbols('x y')

    # Define equations
    equation1 = Eq(x**2 + y**2, 20)
    equation2 = Eq(x**2 - 5*x*y + 6*y**2, 0)

    # Solve the system of equations
    solutions = solve((equation1, equation2), (x, y), dict=True)

    # Return solutions as strings
    return str(solutions)
```
'''
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

"""
    def apply(
        self,
        keywords: Annotated[str,
                            Info('A series of English keywords separated by comma.')],
    ) -> ImageIO:
"""
@mcp_server.tool(
    name="TextToImage",
    description='This tool can generate an image according to the '
                    'input text.'
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

"""
    def apply(self, image: ImageIO, instruction: str) -> ImageIO:
"""
@mcp_server.tool(
    name="ImageStylization",
    description='This tool can modify the input image according to the '
                    'input instruction. Here are some example instructions: '
                    '"turn him into cyborg", "add fireworks to the sky", '
                    '"make his jacket out of leather".'
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
