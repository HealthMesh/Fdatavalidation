import inspect
from types import FunctionType

def decorator(func):
    def wrapper():
        print("before")
        func()
        print("after")
    return wrapper

@decorator
@decorator
def my_function():
    print("func")


print("Decorated:")
my_function()
# Output:
# before
# before
# func
# after
# after


def extract_all_wrapped(decorated):
    wrapped_functions = []
    current_func = decorated

    while current_func.__closure__:
        wrapped_functions.append(current_func)
        # Check closure cells to find the wrapped function
        for cell in current_func.__closure__:
            if isinstance(cell.cell_contents, FunctionType):
                current_func = cell.cell_contents
                break
        else:
            # If no function is found in closure, break out of the loop
            break

    # Add the innermost function
    wrapped_functions.append(current_func)
    return wrapped_functions


# Unwrap and collect all functions from the original to the decorators
wrapped_functions = extract_all_wrapped(my_function)

print("\nSource codes from innermost to outermost:")

# Print the source code of each function
for func in reversed(wrapped_functions):
    print(inspect.getsource(func))
    print("-" * 40)  # Divider between different functions
