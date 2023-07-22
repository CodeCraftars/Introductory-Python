import jax
import jax.numpy as jnp
import numpy as np
import time
from jaxlib.xla_extension import DeviceArray
from cProfile import Profile
import jax.random as random


# Counting  without vectorization
def count_numbers(numbers):
    start = time.time()
    number_counts = {}
    for number in numbers:
        if number not in number_counts:
            number_counts[number] = 1
        else:
            number_counts[number] += 1
    time_taken = time.time() - start
    print(f"total time taken: {time_taken}")
    return number_counts


def count_vectorized(numbers):
    start = time.time()
    unique_numbers, number_counts = jax.numpy.unique(numbers, return_counts=True)
    number_counts_dict = dict(zip(unique_numbers, number_counts))
    time_taken = time.time() - start
    print(f"total time taken: {time_taken}")
    return number_counts_dict


def multiply_matrices(matrix1, matrix2, matrix3, matrix4):
    return matrix1 * matrix2 * matrix3 * matrix4

def create_random_array(shape: tuple, seed: int) -> DeviceArray:
    prng_key = random.PRNGKey(seed)
    random_numbers: DeviceArray = random.uniform(prng_key, shape)
    return random_numbers

if __name__ == "__main__":
    # Create input matrices
    matrix_batch = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    
    start_vmap = time.time()
    # Using jax.vmap
    vectorized_exponential = jax.vmap(jnp.exp)
    result_vmap = vectorized_exponential(matrix_batch)
    total_time_vmap = time.time()-start_vmap 

    start_exp = time.time()
    # Using basic JAX functions
    result_basic = jnp.exp(matrix_batch)
    total_time_exp = time.time() - start_exp

    print("Result using jax.vmap:\n", total_time_vmap, result_vmap)
    print("Result using basic JAX functions:\n", total_time_exp, result_basic)


# if __name__ == "__main__":
    
#     # Create input matrices
#     start  = time.time()
#     matrix_batch1 = create_random_array((1000000, 2), 0)
#     matrix_batch2 = create_random_array((1000000, 2), 1)
#     matrix_batch3 = create_random_array((1000000, 2), 2)
#     matrix_batch4 = create_random_array((1000000, 2), 3)
#     # Apply jax.vmap to vectorize the function across the first axis
#     vectorized_multiply_matrices = jax.vmap(multiply_matrices, in_axes=(0, 0, 0, 0))

#     # Perform element-wise matrix multiplication on the batch of matrices
#     result = vectorized_multiply_matrices(matrix_batch1, matrix_batch2, matrix_batch3, matrix_batch4)
#     total_time = time.time()-start
#     print(total_time)
#     print(result)


# if __name__ == "__main__":
#     # profile = Profile()
#     # profile.enable()
#     numbers = np.random.randint(10, size=1000000)
#     func = count_numbers
#     result = jax.vmap(func)(numbers)
#     print(f"result of {func.__name__}: {result}" )
#     # profile.disable()
#     # profile.print_stats()
