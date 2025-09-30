import sys, os
sys.path.append(os.path.abspath("../agentlego"))

from agentlego.tools import ImageStylization
from agentlego.types import ImageIO

def test_image_stylization():
    tool = ImageStylization(device='cpu') 
    tool.setup()

    image_path = "../gta_dataset/image/image_1.jpg"  
    image = ImageIO(image_path)

    style = "add fireworks to the sky"  

    result = tool.apply(image=image, instruction=style)

    # svae image
    result.to_pil().save("./images/stylized_image.jpg")
    print("Stylized image saved to stylized_image.jpg")

if __name__ == "__main__":
    test_image_stylization()