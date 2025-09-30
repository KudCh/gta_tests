import sys, os
sys.path.append(os.path.abspath("../agentlego"))

from agentlego.tools import Calculator

def test_calculator():
    # Initialize the Calculator tool
    tool = Calculator()
    tool.setup()

    # Define the mathematical expression
    expression = "12 / 4 + 3 * (2 - 1)"

    # Apply the tool
    result = tool.apply(expression)

    # Print the result for verification
    print("Calculator Result:", result)

if __name__ == "__main__":
    test_calculator()