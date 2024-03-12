import numpy as np

def d_max_relative_entropy(rho, sigma):
    tol = 1e-7

    # Check support condition
    rvals, rvecs = np.linalg.eigh(rho)
    if any(abs(rvals.imag) >= tol):
        raise ValueError("Input rho has non-real eigenvalues.")
    rvals = rvals.real
    svals, svecs = np.linalg.eigh(sigma)
    if any(abs(svals.imag) >= tol):
        raise ValueError("Input sigma has non-real eigenvalues.")
    svals = svals.real

    # Calculate inner products of eigenvectors and return +inf if kernel
    # of sigma overlaps with support of rho.
    P = abs(rvecs.T @ svecs.conj()) ** 2
    if (rvals >= tol) @ (P >= tol) @ (svals < tol):
        return np.inf

    # If true
    s = np.log2(np.max(np.linalg.eigvals(np.linalg.inv(sigma) @ rho)))
    return s