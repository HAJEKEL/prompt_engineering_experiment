import os
import re
import json
import uuid
import logging
import commentjson
import numpy as np

class StiffnessMatrixProcessor:
    """
    A class to process stiffness matrices:
    - Extracts stiffness matrices from a response string.
    - Validates and saves the matrix to a file.
    """

    def __init__(self, matrices_dir='matrices'):
        """
        Initializes the processor with a specified directory for saving matrices.

        Parameters:
            matrices_dir (str): Directory where matrices are saved.
        """
        self.matrices_dir = matrices_dir
        self.ensure_directories()

    def ensure_directories(self):
        """
        Ensures that the required directories exist.
        """
        os.makedirs(self.matrices_dir, exist_ok=True)

    def extract_stiffness_matrix(self, response):
        """
        Extracts the stiffness matrix from a response string and saves it to a file.

        Parameters:
            response (str): The response string containing the stiffness matrix in a JSON code block.

        Returns:
            tuple: A tuple containing the stiffness matrix and the path to the saved matrix file,
                   or (None, None) if extraction fails.
        """
        # Define the pattern to extract the JSON code block
        pattern = r"```json\s*\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)

        if not match:
            logging.error("No JSON code block found in the response.")
            return None, None

        json_code = match.group(1)

        # Attempt to parse JSON with commentjson
        try:
            data = commentjson.loads(json_code)
        except Exception as e:
            logging.warning(f"Failed to parse JSON code block: {e}")
            return None, None

        stiffness_matrix = data.get('stiffness_matrix')
        if stiffness_matrix is None:
            logging.error("Key 'stiffness_matrix' not found in JSON data.")
            return None, None

        # Validate matrix structure
        if not self.validate_stiffness_matrix(stiffness_matrix):
            return None, None

        # Check if all elements are numeric
        if not self.is_numeric_matrix(stiffness_matrix):
            logging.warning("Stiffness matrix contains non-numeric values. Skipping extraction.")
            return None, None

        # Save stiffness matrix to a file with unique naming
        matrix_filename = f"matrix_{self.generate_unique_id()}.json"
        matrix_file_path = os.path.join(self.matrices_dir, matrix_filename)

        try:
            with open(matrix_file_path, "w", encoding="utf-8") as matrix_file:
                json.dump(stiffness_matrix, matrix_file, indent=2)
            logging.info(f"Stiffness matrix extracted and saved at {matrix_file_path}: {stiffness_matrix}")
            return stiffness_matrix, matrix_file_path
        except Exception as e:
            logging.error(f"Failed to save stiffness matrix to file: {e}")
            return None, None

    def validate_stiffness_matrix(self, matrix):
        """
        Validates that the stiffness matrix is a 3x3 matrix.

        Parameters:
            matrix (list): The stiffness matrix to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        if not isinstance(matrix, list) or len(matrix) != 3:
            logging.error(f"Expected stiffness_matrix to be a list of 3 lists, found {type(matrix)} with length {len(matrix) if isinstance(matrix, list) else 'N/A'}.")
            return False

        for i, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != 3:
                logging.error(f"Expected row {i} to be a list of 3 elements, found {type(row)} with length {len(row) if isinstance(row, list) else 'N/A'}.")
                return False

        return True

    def is_numeric_matrix(self, matrix):
        """
        Checks if all elements in the stiffness matrix are numeric.

        Parameters:
            matrix (list): The stiffness matrix to check.

        Returns:
            bool: True if all elements are numeric, False otherwise.
        """
        for row in matrix:
            for element in row:
                if not isinstance(element, (int, float)):
                    logging.warning(f"Non-numeric element detected: {element}")
                    return False
        return True

    def generate_unique_id(self):
        """
        Generates a unique identifier for naming files.

        Returns:
            str: A unique identifier string.
        """
        return uuid.uuid4().hex

if __name__ == "__main__":
    processor = StiffnessMatrixProcessor()

    # Sample valid response
    sample_response = """Certainly! Here is the stiffness matrix with a high stiffness of 1000 N/m in the X direction and a stiffness of 200 N/m in the Y and Z directions:

### Stiffness Matrix
```json
{
  "stiffness_matrix": [
    [1000, 0, 0],
    [0, 200, 0],
    [0, 0, 200]
  ]
}
```"""

    stiffness_matrix, matrix_path = processor.extract_stiffness_matrix(sample_response)
    if stiffness_matrix:
        print(f"Stiffness Matrix saved at: {matrix_path}")
    else:
        print("Failed to extract stiffness matrix.")

    # Test with placeholders
    bad_response = """Certainly! 
### Stiffness Matrix
```json
{
  "stiffness_matrix": [
    [K_high, 0, 0],
    [0, K_low, 0],
    [0, 0, K_low]
  ]
}
```"""

    matrix, path = processor.extract_stiffness_matrix(bad_response)
    # => Should return (None, None) and log a warning about placeholders.
    if matrix:
        print(f"Stiffness Matrix saved at: {path}")
    else:
        print("Failed to extract stiffness matrix.")
