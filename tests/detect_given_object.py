import sys, os
sys.path.append(os.path.abspath("../agentlego"))

from agentlego.tools import ObjectDetection 
from agentlego.types import ImageIO

def test_detect_given_object():
    # Initialize the DetectGivenObject tool
    tool = ObjectDetection()
    tool.setup()

    # Load a test image
    image_path = "../gta_dataset/image/image_1.jpg"  # Replace with your test image path
    image = ImageIO(image_path)

    # Define the object to detect
    object = "beer"  
    result = tool.apply(image)

    # Parse and filter results
    lines = str(result).splitlines()
    print("All detected objects:", lines)
    
    filtered = [line for line in lines if line.lower().startswith(object.lower())]
    if not filtered:
        return f"No instances of '{object}' found."
    return "Object detection results: " + "\n".join(filtered)

if __name__ == "__main__":
    print(test_detect_given_object())