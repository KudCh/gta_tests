import sys, os
sys.path.append(os.path.abspath("../agentlego"))

from agentlego.tools import ImageDescription
from agentlego.types import ImageIO

def test_image_description():
    tool = ImageDescription()
    tool.setup()

    image_path = "../gta_dataset/image/image_1.jpg"  
    image = ImageIO(image_path)

    result = tool.apply(image=image)

    print("Result:", result)

if __name__ == "__main__":
    test_image_description()