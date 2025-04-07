# Einops_implementation

A simple implementation of the Einops library [https://einops.rocks/] on a single colab notebook, with added test cases.

## Approach
The `EinopsManan` class implements the following steps:

1.  **Parsing the Pattern (`parse_lr`)**: The input pattern string is split into left and right sides, and each side is tokenized into individual dimension names, grouped dimensions (within parentheses), and the ellipsis.

2.  **Expanding Dimensions (`expand_left_and_right_sides`)**: The token lists are expanded to handle the ellipsis and grouped dimensions. The ellipsis is replaced with placeholder dimension names based on the input tensor's shape. Grouped dimensions are split into individual dimension names in the expanded lists.

3.  **Creating Index Mappings (`create_idx_mapping`)**: Dictionaries are created to map the expanded dimension names on the left and right sides to their respective indices.

4.  **Extracting Bracketed Groups (`get_brackets`)**: The right-side tokens are examined to identify grouped dimensions (within parentheses). The indices of these grouped dimensions in the expanded right-side list are stored.

5.  **Generating the Resulting Array (`get_resulting_array`)**: This is the core transformation logic:
    * Reshape: The input tensor is reshaped based on the left-side pattern and any provided `axes_lengths`. Grouped dimensions on the left side might require calculating the size of one dimension if the total size and other group member sizes are known.
    * Transpose: The reshaped tensor is transposed to match the order of dimensions specified by the right-side pattern, considering the dimensions that are common to both left and right sides.
    * Handle New Dimensions: Dimensions present in the right side but not the left are treated as new dimensions. Their sizes must be provided in `axes_lengths`, and they are inserted into the tensor with appropriate repetition.
    * Handle Squeezed Dimensions: Dimensions present in the left side but not the right are expected to have a size of 1 and are squeezed out.
    * Handle Bracketed Output Dimensions: Dimensions grouped in the right-side pattern are reshaped into a single dimension in the output tensor.

6.  **Rearrange (`rearrange`)**: This is the main user-facing function that orchestrates all the above steps to perform the tensor rearrangement.


## How to Run

You can run the cells directly in the colab notebook, and add any testcases in the following format:
result = einops.rearrange(your_array,pattern, split/group variable dimensions)



Endnote: This is bound to have testcases that might not have passed because of the implementation being done in a span of 2 days only, happy to accept issues.
