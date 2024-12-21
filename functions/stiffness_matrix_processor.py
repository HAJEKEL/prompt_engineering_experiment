import os
import re
import json
import uuid
import logging
import numpy as np
import matplotlib.pyplot as plt
import commentjson

class StiffnessMatrixProcessor:
    """
    A class to process stiffness matrices:
    - Extracts stiffness matrices from a response string.
    - Validates and saves the matrix to a file.
    """

    def __init__(
        self,
        matrices_dir='matrices'):
        """
        Initializes the processor with optional base URLs and directories.

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
            tuple: A tuple containing the stiffness matrix and the URL to the saved matrix file.
        """
        # Define the pattern to extract the JSON code block
        pattern = r"```json\n(.*?)\n```"
        match = re.search(pattern, response, re.DOTALL)

        if not match:
            logging.error("No JSON code block found in the response.")
            return None, None

        json_code = match.group(1)

        try:
            # Parse the JSON code with comments
            data = commentjson.loads(json_code)
            stiffness_matrix = data.get('stiffness_matrix')

            if stiffness_matrix is None:
                logging.error("Key 'stiffness_matrix' not found in JSON data.")
                return None, None

            # Validate the stiffness matrix structure
            if not self.validate_stiffness_matrix(stiffness_matrix):
                return None, None

            # Save stiffness matrix to a file
            matrix_filename = f"{uuid.uuid4()}.json"
            matrix_file_path = os.path.join(self.matrices_dir, matrix_filename)

            with open(matrix_file_path, "w") as matrix_file:
                json.dump(stiffness_matrix, matrix_file)
            logging.info(f"Stiffness matrix extracted and saved at {matrix_file_path}: {stiffness_matrix}")

            return stiffness_matrix

        except commentjson.JSONLibraryException as e:
            logging.error(f"Error parsing JSON with comments: {e}")
            return None, None

    def validate_stiffness_matrix(self, matrix):
        """
        Validates that the stiffness matrix is a 3x3 matrix.

        Parameters:
            matrix (list): The stiffness matrix to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        if len(matrix) != 3:
            logging.error(f"Expected 3 rows in stiffness_matrix, found {len(matrix)}.")
            return False

        for i, row in enumerate(matrix):
            if len(row) != 3:
                logging.error(f"Expected 3 values in row {i}, found {len(row)}.")
                return False

        return True

if __name__ == "__main__":
    # To switch between localhost and public URLs, set use_public_urls accordingly
    processor = StiffnessMatrixProcessor()

    # Sample response containing the stiffness matrix in a JSON code block
    sample_response ="Certainly! Here is the stiffness matrix with a high stiffness of 1000 N/m in the X direction and a stiffness of 200 N/m in the Y and Z directions:\n\n### Stiffness Matrix\n```json\n{\n  \"stiffness_matrix\": [\n    [1000, 0, 0],\n    [0, 200, 0],\n    [0, 0, 200]\n  ]\n}\n```"


    # Extract the stiffness matrix from the sample response
    stiffness_matrix, matrix_url = processor.extract_stiffness_matrix(sample_response)
    if stiffness_matrix:
        print(f"Stiffness Matrix URL: {matrix_url}")

        # Generate the ellipsoid plot
        ellipsoid_url = processor.generate_ellipsoid_plot(stiffness_matrix)
        if ellipsoid_url:
            print(f"Ellipsoid Plot URL: {ellipsoid_url}")
