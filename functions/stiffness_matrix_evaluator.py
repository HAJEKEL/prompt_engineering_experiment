import numpy as np

class StiffnessMatrixEvaluator:
    """
    A simplified evaluator that returns True if the
    estimated 3×3 matrix matches the ground truth (within
    a small numerical tolerance), or False otherwise.
    """
    def __init__(self, rtol=1e-5, atol=1e-8):
        """
        Args:
            rtol (float): Relative tolerance for floating-point comparisons.
            atol (float): Absolute tolerance for floating-point comparisons.
        """
        self.rtol = rtol
        self.atol = atol

    def evaluate_stiffness_matrix(self, gt_matrix, estimated_matrix):
        """
        Returns a dict with a single boolean key 'correct':
         - True if the 3×3 match (within tolerance).
         - False otherwise.
        """
        # Convert to numpy arrays just in case they aren't already
        gt = np.array(gt_matrix, dtype=float)
        est = np.array(estimated_matrix, dtype=float)

        # Check shape first (must be 3×3)
        if gt.shape != (3, 3) or est.shape != (3, 3):
            return {"correct": False}

        # Compare element-wise within specified tolerance
        are_close = np.allclose(gt, est, rtol=self.rtol, atol=self.atol)
        return {"correct": are_close}
