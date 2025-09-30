import sys, os
sys.path.append(os.path.abspath("../agentlego"))
from agentlego.tools import Solver

def test_solver():
    # Initialize the Solver tool
    tool = Solver()

    import textwrap
    # Define python command
    command = '''
        ```python
        # import packages
        from sympy import symbols, Eq, solve
        def solution():
            # Define symbols
            x, y = symbols('x y')

            # Define equations
            equation1 = Eq(x**2 + y**2, 20)
            equation2 = Eq(x**2 - 5*x*y + 6*y**2, 0)

            # Solve the system of equations
            solutions = solve((equation1, equation2), (x, y), dict=True)

            # Return solutions as strings
            return str(solutions)
        ```
        '''
    command = textwrap.dedent(command).strip()
   
    # Apply the tool
    result = tool.apply(command=command)

    # Print the result for verification
    print("Solver Result:", result)

if __name__ == "__main__":
    test_solver()