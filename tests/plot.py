import sys, os
sys.path.append(os.path.abspath("../agentlego"))

from agentlego.tools import Plot
from agentlego.types import ImageIO

def test_plot():
    # Initialize the Plot tool
    tool = Plot()

    import textwrap
    # Define python command
    command = '''
        ```python
        # import packages
        import matplotlib.pyplot as plt
        def solution():
            # labels and data
            cars = ['AUDI', 'BMW', 'FORD', 'TESLA', 'JAGUAR', 'MERCEDES']
            data = [23, 17, 35, 29, 12, 41]

            # draw diagrams
            figure = plt.figure(figsize=(8, 6))
            plt.pie(data, labels=cars, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')
            plt.title('Car Distribution')
            return figure
        ```
        '''
    command = textwrap.dedent(command).strip()
   
    # Apply the tool
    result_image = tool.apply(command=command)

    # Save or display the result for verification
    result_image.to_pil().save("./images/plot.png")
    print("Plot created and saved to plot.jpg")

if __name__ == "__main__":
    test_plot()