import sys, os
sys.path.append(os.path.abspath("../agentlego"))
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv()) 

from agentlego.tools import MathOCR
from agentlego.types import ImageIO

def test_math_ocr():
    # Initialize the MathOCR tool
    tool = MathOCR()

    # Load a test image
    image_path = "./images/math_expression.jpg"  
    image = ImageIO(image_path)

    # Apply the tool
    result = tool.apply(image=image)

    # Print the result for verification
    print("Math OCR Result:", result)

if __name__ == "__main__":
    test_math_ocr()