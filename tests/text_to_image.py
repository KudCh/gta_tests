import sys, os
sys.path.append(os.path.abspath("../agentlego"))

from agentlego.tools import TextToImage

def test_text_to_image():
    # Initialize the TextToImage tool
    tool = TextToImage(device='cpu')  # Use 'cuda' if a GPU is available
    tool.setup()

    # Define the text prompt
    prompt = "A futuristic cityscape with flying cars and neon lights"
    keywords = ", ".join(prompt.split())

    # Apply the tool
    result = tool.apply(keywords=keywords)

    # Save the generated image
    result.to_pil().save("./images/text_to_image.png") 
    print("Image generated and saved to text_to_image.png")

if __name__ == "__main__":
    test_text_to_image()