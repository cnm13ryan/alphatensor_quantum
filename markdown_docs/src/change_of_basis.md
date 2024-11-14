## FunctionDef _sample_triangular_matrix(rng, size, upper_triangular, prob_zero_entry)
**Function Overview**: The `_sample_triangular_matrix` function generates a random triangular matrix with diagonal entries set to 1.

**Parameters**:
- **rng**: A Jax random key used to generate random numbers. 
  - **referencer_content**: True (Referenced by `generate_change_of_basis`)
  - **reference_letter**: False
- **size**: An integer representing the size of the matrix.
  - **referencer_content**: True (Referenced by `generate_change_of_basis`)
  - **reference_letter**: False
- **upper_triangular**: A boolean indicating whether the matrix should be upper triangular (`True`) or lower triangular (`False`).
  - **referencer_content**: True (Referenced by `generate_change_of_basis`)
  - **reference_letter**: False
- **prob_zero_entry**: A float representing the probability that an off-diagonal entry will be zero.
  - **referencer_content**: True (Referenced by `generate_change_of_basis`)
  - **reference_letter**: False

**Return Values**:
- Returns a triangular matrix of size `size x size` with diagonal entries set to 1, and off-diagonal entries randomly sampled based on the provided probability.

**Detailed Explanation**:
The `_sample_triangular_matrix` function constructs a triangular matrix by first sampling random binary values (0 or 1) for its off-diagonal elements using a Bernoulli distribution. The `jax.random.bernoulli` function is used to generate these random values, where the probability of an entry being zero is given by `prob_zero_entry`. 

The `masking_fn` variable selects either `jnp.triu` or `jnp.tril` based on whether the matrix should be upper triangular or lower triangular. The selected masking function then ensures that only the appropriate triangle (upper or lower) of the matrix contains non-zero values, while the other part is set to zero.

Finally, an identity matrix of the same size is added to the sampled triangle matrix. This addition ensures that all diagonal entries are 1, which guarantees that the determinant of the resulting matrix is 1 and hence it is invertible.

**Relationship Description**:
The `_sample_triangular_matrix` function is called by `generate_change_of_basis`, which uses this function to generate both an upper triangular and a lower triangular matrix. These matrices are then multiplied together modulo 2 to produce a final change of basis matrix that is guaranteed to be invertible.

**Usage Notes and Refactoring Suggestions**:
- **Limitations**: The function assumes the input parameters are correctly formatted (e.g., `size` should be a positive integer, `prob_zero_entry` should be in the range [0, 1]). Adding input validation could improve robustness.
- **Edge Cases**: When `prob_zero_entry` is very close to 1, the resulting matrix may have many zero entries, which might not be desirable for certain applications requiring more dense matrices. Consider adding a check or warning if this probability threshold is exceeded.
- **Refactoring Suggestions**:
  - **Extract Method**: The logic for creating the triangular part of the matrix could be extracted into a separate method to improve readability and modularity.
  - **Introduce Explaining Variable**: For clarity, introduce explaining variables for complex expressions such as `masking_fn` or the addition of the identity matrix.
  - **Simplify Conditional Expressions**: The conditional logic for selecting between `jnp.triu` and `jnp.tril` could be simplified by using a dictionary or another method to map boolean values directly to functions, reducing the need for an explicit if-else statement.

By applying these refactoring techniques, the code can become more readable, maintainable, and easier to extend in the future.
## FunctionDef generate_change_of_basis(size, prob_zero_entry, rng)
**Function Overview**: The `generate_change_of_basis` function generates a change of basis matrix that is guaranteed to be invertible.

**Parameters**:
- **size**: An integer representing the desired size of the matrix.
  - **referencer_content**: True (Referenced by other components within the project)
  - **reference_letter**: False
- **prob_zero_entry**: A float representing the probability of the sampled entries being zero.
  - **referencer_content**: True (Referenced by other components within the project)
  - **reference_letter**: False
- **rng**: A Jax random key used to generate random numbers.
  - **referencer_content**: True (Referenced by other components within the project)
  - **reference_letter**: False

**Return Values**:
- Returns a change of basis matrix of size `size x size`, with entries being either 0 or 1, and is guaranteed to be invertible.

**Detailed Explanation**:
The `generate_change_of_basis` function begins by splitting the provided random number generator (`rng`) into two separate generators. It then uses these generators to create two triangular matrices through calls to `_sample_triangular_matrix`. The first call creates an upper triangular matrix with a probability of zero entries given by `prob_zero_entry`, and the second call creates a lower triangular matrix under similar conditions. These two matrices are added together, and the result is taken modulo 2 to ensure all elements are either 0 or 1. This process guarantees that the resulting matrix is invertible.

**Relationship Description**:
The function relies on `_sample_triangular_matrix` for generating triangular matrices with specified properties. It does not have any documented references from other parts of the project as callers, but it is referenced by other components within the project as a callee.

**Usage Notes and Refactoring Suggestions**:
- **Limitations**: The function assumes that combining two triangular matrices and taking modulo 2 will always result in an invertible matrix. While this holds true for the given implementation, it may not be immediately obvious to someone reading the code.
- **Edge Cases**: When `prob_zero_entry` is very high (close to 1), the generated matrix might have a higher chance of being singular or nearly singular before applying modulo 2 operation. However, due to the nature of the operations involved, the final matrix should still be invertible as per the logic.
- **Refactoring Suggestions**:
  - **Extract Method**: The creation and combination of upper and lower triangular matrices could be extracted into a separate method for better readability and modularity.
  - **Introduce Explaining Variable**: Introducing variables to hold intermediate results, such as the generated upper and lower triangular matrices, can improve clarity.
  - **Simplify Conditional Expressions**: If additional conditions or modifications are needed in future development, consider using guard clauses to simplify conditional logic.

By applying these refactoring techniques, the code can become more readable, maintainable, and easier to extend in the future.
## FunctionDef apply_change_of_basis(tensor, cob_matrix)
**Function Overview**: The `apply_change_of_basis` function applies a change of basis transformation to a given tensor using a specified change of basis matrix.

**Parameters**:
- **tensor**: The input tensor that needs to be transformed. It is expected to have the shape `(size, size, size)`.
  - **referencer_content**: Not explicitly provided in the documentation.
  - **reference_letter**: Not explicitly provided in the documentation.
- **cob_matrix**: The change of basis matrix used for transforming the tensor. This matrix should have the shape `(size, size)`.

**Return Values**:
- The function returns a transformed tensor with the same shape as the input tensor, i.e., `(size, size, size)`. Each element in the returned tensor is computed using the provided change of basis matrix and then taken modulo 2 to ensure binary values.

**Detailed Explanation**:
The `apply_change_of_basis` function performs a transformation on a three-dimensional tensor using a specified change of basis matrix. The core operation involves applying the change of basis matrix along each dimension of the tensor. This is achieved through the use of `jnp.einsum`, which allows for concise and efficient computation of Einstein summation convention expressions.

The expression `'ia,jb,kc,abc->ijk'` in `jnp.einsum` specifies how the indices of the input arrays are contracted to produce the output array. Here:
- `ia` indicates that the first dimension of the tensor is transformed using the change of basis matrix.
- `jb` indicates that the second dimension of the tensor is transformed using the same change of basis matrix.
- `kc` indicates that the third dimension of the tensor is also transformed using the same change of basis matrix.
- `abc->ijk` specifies how the resulting indices from the contractions are mapped to form the output tensor.

After the transformation, each element in the resultant tensor is taken modulo 2. This operation ensures that all values in the final tensor are binary (either 0 or 1).

**Relationship Description**:
- Since neither `referencer_content` nor `reference_letter` are provided and truthy, there is no functional relationship to describe between this function and other components within the project based on the given information.

**Usage Notes and Refactoring Suggestions**:
- **Limitations**: The function assumes that both the tensor and the change of basis matrix have compatible shapes. If these assumptions are not met, the function will raise an error.
- **Edge Cases**: Consider edge cases where the input tensor or cob_matrix contains non-integer values or negative numbers, as the modulo operation may yield unexpected results in such scenarios.
- **Refactoring Suggestions**:
  - **Introduce Explaining Variable**: The `jnp.einsum` expression can be complex for some readers. Introducing an intermediate variable to store the result of the einsum operation could improve readability.
    ```python
    transformed_tensor = jnp.einsum('ia,jb,kc,abc->ijk', cob_matrix, cob_matrix, cob_matrix, tensor)
    binary_transformed_tensor = jnp.mod(transformed_tensor, 2)
    return binary_transformed_tensor
    ```
  - **Extract Method**: If the function grows in complexity or if similar transformations are needed elsewhere, consider extracting the transformation logic into a separate method.
    ```python
    def transform_tensor_with_cob(tensor, cob_matrix):
        return jnp.einsum('ia,jb,kc,abc->ijk', cob_matrix, cob_matrix, cob_matrix, tensor)

    def apply_change_of_basis(tensor, cob_matrix):
        transformed_tensor = transform_tensor_with_cob(tensor, cob_matrix)
        return jnp.mod(transformed_tensor, 2)
    ```
- **Encapsulate Collection**: If the function is part of a larger system where tensors and matrices are frequently manipulated, consider encapsulating these operations within classes to improve modularity and maintainability.
