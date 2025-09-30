import sys, os
sys.path.append(os.path.abspath("../agentlego"))

from agentlego.tools import AddText
from agentlego.types import ImageIO

def test_add_text():
    # Initialize the AddText tool
    tool = AddText()

    # Load a test image
    image_path = "../gta_dataset/image/image_1.jpg"  # Replace with your test image path
    image = ImageIO(image_path)

    # Define text and position
    text = "Test Text"
    position = "lt"  # left-top corner
    color = "blue"

    # Apply the tool
    result_image = tool.apply(image=image, text=text, position=position, color=color)

    # Save or display the result for verification
    result_image.to_pil().save("./images/add_text.jpg")
    print("Text added and saved to add_text.jpg")

if __name__ == "__main__":
    test_add_text()