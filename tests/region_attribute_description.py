import torch
from PIL import Image
import sys, os
sys.path.append(os.path.abspath("../agentlego"))

from benchmark import RegionAttributeDescriptionReimplemented
from agentlego.types import ImageIO

def test_region_attribute_description():
    tool = RegionAttributeDescriptionReimplemented(device="cpu")
    tool.setup()

    image_path = "../gta_dataset/image/image_1.jpg"  
    image = ImageIO(image_path)

    bbox = "50, 50, 200, 200"  
    attribute = "color"

    result = tool(image=image, bbox=bbox, attribute=attribute)

    print("Result:", result)

if __name__ == "__main__":
    test_region_attribute_description()
