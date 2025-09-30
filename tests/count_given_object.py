import sys, os
sys.path.append(os.path.abspath("../agentlego"))

from agentlego.tools import CountGivenObject
from agentlego.types import ImageIO

import nltk
nltk.data.path.append("./~/nltk_data")

def test_count_given_object():
    # Initialize the CountGivenObject tool
    tool = CountGivenObject()
    tool.setup()

    # Load a test image
    image_path = "../gta_dataset/image/image_1.jpg"  # Replace with your test image path
    image = ImageIO(image_path)

    # Define the object to count
    object_name = "beer bottle"  

    # Apply the tool
    count = tool.apply(image=image, text=object_name)

    # Print the result for verification
    print(f"Number of '{object_name}' in the image:", count)

if __name__ == "__main__":
    test_count_given_object()