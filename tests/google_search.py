import sys, os
sys.path.append(os.path.abspath("../agentlego"))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from agentlego.tools import GoogleSearch

def test_google_search():
    # Initialize the GoogleSearch tool
    tool = GoogleSearch()
    tool.setup()

    # Define the search query
    query = "What is the weather in Paris?"

    # Apply the tool
    result = tool.apply(query)

    # Print the result for verification
    print("Google Search Result:", result)

if __name__ == "__main__":
    test_google_search()