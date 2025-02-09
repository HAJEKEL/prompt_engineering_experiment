import numpy as np

class StiffnessMatrixEvaluator:
    """
    A class that evaluates a 3x3 stiffness matrix against a ground-truth matrix
    using various metrics (MAE, MSE, Frobenius norm, eigenvalue/eigenvector analysis).
    """
    
    def __init__(self):
        pass

    def evaluate_stiffness_matrix(self, gt_matrix, estimated_matrix):
        """
        Evaluates the stiffness matrix using multiple metrics:
          1. Element-wise Error (MAE, MSE).
          2. Frobenius Norm Difference.
          3. Angular Deviation (between eigenvectors).
          4. Eigenvalue Magnitude Difference.

        Args:
            gt_matrix (list or np.ndarray): The ground-truth stiffness matrix (3x3).
            estimated_matrix (list or np.ndarray): The predicted/estimated matrix (3x3).

        Returns:
            dict: A dictionary containing numerical metrics.
        """
        # Ensure we have numpy arrays
        gt_matrix = np.array(gt_matrix, dtype=float)
        estimated_matrix = np.array(estimated_matrix, dtype=float)

        # 1. Mean Absolute Error (MAE) & Mean Squared Error (MSE)
        mae = np.mean(np.abs(gt_matrix - estimated_matrix))
        mse = np.mean((gt_matrix - estimated_matrix) ** 2)

        # 2. Frobenius Norm Difference
        fro_diff = np.linalg.norm(gt_matrix - estimated_matrix, ord='fro')

        # 3. Angular Deviation Between Eigenvectors
        gt_evals, gt_evecs = np.linalg.eigh(gt_matrix)
        est_evals, est_evecs = np.linalg.eigh(estimated_matrix)

        # Normalize eigenvectors column-wise
        gt_evecs_norm = gt_evecs / np.linalg.norm(gt_evecs, axis=0)
        est_evecs_norm = est_evecs / np.linalg.norm(est_evecs, axis=0)

        # Dot products for each column (principal axis)
        dot_products = np.abs(np.sum(gt_evecs_norm * est_evecs_norm, axis=0))
        dot_products_clipped = np.clip(dot_products, -1.0, 1.0)
        angular_deviation = np.arccos(dot_products_clipped)

        # 4. Eigenvalue Magnitude Difference
        gt_evals_sorted = np.sort(gt_evals)
        est_evals_sorted = np.sort(est_evals)
        eval_diff = np.abs(gt_evals_sorted - est_evals_sorted)

        return {
            "mae": mae,
            "mse": mse,
            "frobenius_norm_diff": fro_diff,
            "angular_deviation": angular_deviation.tolist(),
            "eigenvalue_magnitude_diff": eval_diff.tolist()
        }
