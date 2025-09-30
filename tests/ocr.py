import sys, os
sys.path.append(os.path.abspath("../agentlego"))

from agentlego.tools import OCR
from agentlego.types import ImageIO

def test_ocr():
    # Initialize the OCR tool
    tool = OCR()
    tool.setup()

    # Load a test image
    image_path = "../gta_dataset/image/image_2.jpg"  # Replace with your test image path
    image = ImageIO(image_path)

    # Apply the tool
    result = tool.apply(image=image)

    # Print the result for verification
    print("OCR Result:", result)

if __name__ == "__main__":
    test_ocr()