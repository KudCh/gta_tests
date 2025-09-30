import sys, os
sys.path.append(os.path.abspath("../agentlego"))

from agentlego.tools import DrawBox
from agentlego.types import ImageIO

def test_draw_box():
    # Initialize the DrawBox tool
    tool = DrawBox()

    # Load a test image
    image_path = "../gta_dataset/image/image_1.jpg"  # Replace with your test image path
    image = ImageIO(image_path)

    # Define bounding box (x1, y1, x2, y2)
    bbox = "50, 50, 200, 200"  # Example coordinates
    color = "red"
    thickness = 3

    # Apply the tool
    result_image = tool.apply(image=image, bbox=bbox, annotation="Test Box")

    # Save or display the result for verification
    result_image.to_pil().save("./images/draw_box.jpg")
    print("Box drawn and saved to draw_box.jpg")

if __name__ == "__main__":
    test_draw_box()