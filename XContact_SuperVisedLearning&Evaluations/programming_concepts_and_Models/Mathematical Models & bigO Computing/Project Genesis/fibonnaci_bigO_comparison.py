import time


# Jiaoe's original brute force recursive function
def jiaoe_fibonacci(n):
    if n <= 1:
        return n
    else:
        return jiaoe_fibonacci(n - 1) + jiaoe_fibonacci(n - 2)


def gemini_fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return gemini_fibonacci(n - 1) + gemini_fibonacci(n - 2)


# Morpheus function using memoization (more efficient)
def morpheus_fibonacci(n, memo=None):
    if memo is None:
        memo = {0: 0, 1: 1}
    if n not in memo:
        memo[n] = morpheus_fibonacci(n - 1, memo) + morpheus_fibonacci(n - 2, memo)
    return memo[n]


# Function to measure execution time of a Fibonacci function
def measure_function_execution_time(func, n):
    start_time = time.time()
    result = func(n)
    end_time = time.time()
    return result, end_time - start_time


# Testing both functions and comparing execution times
n_input = 33  # Input number for comparison - calculate the N-th Fibonacci Number

# Measure execution time of Jiaoe's brute force function
jiaoe_result, jiaoe_time = measure_function_execution_time(jiaoe_fibonacci, n_input)

# Measure execution time of Morpheus' memoized function
morpheus_result, morpheus_time = measure_function_execution_time(morpheus_fibonacci, n_input)

# Print the results and execution times
print(f"Jiaoe's Fibonacci Result (n={n_input}): {jiaoe_result}")
print(f"Jiaoe's Execution Time: {jiaoe_time:.6f} seconds\n")

print(f"Morpheus' Fibonacci Result (n={n_input}): {morpheus_result}")
print(f"Morpheus' Execution Time: {morpheus_time:.6f} seconds\n")



# Measure execution time of Jiaoe's brute force function
g_result, g_time = measure_function_execution_time(gemini_fibonacci, n_input)

print(f"G's Fibonacci Result (n={n_input}): {jiaoe_result}")
print(f"G's Execution Time: {jiaoe_time:.6f} seconds\n")




