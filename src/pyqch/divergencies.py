import numpy as np

def hockey_stick(rho, sigma, gamma):
    return 1/2 * np.linalg.norm(rho- gamma * sigma, 'nuc')+1/2*np.trace(rho-gamma*sigma)


def hs_dist(r, s):
    return np.linalg.norm(r-s, 'fro')


def max_relative_entropy(rho, sigma, basis=2):
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
    s = np.log2(np.max(np.linalg.eigvals(np.linalg.pinv(sigma) @ rho))) / np.log2(basis)
    return s


def relative_entropy(rho, sigma, basis=2):
    tol = 1e-7
    # S(rho || sigma) = sum_i(p_i log p_i) - sum_ij(p_i P_ij log q_i)
    #
    # S is +inf if the kernel of sigma (i.e. svecs[svals == 0]) has non-trivial
    # intersection with the support of rho (i.e. rvecs[rvals != 0]).
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
    # Avoid -inf from log(0) -- these terms will be multiplied by zero later
    # anyway
    svals[abs(svals) < tol] = 1
    nzrvals = rvals[abs(rvals) >= tol]
    # Calculate S
    S = (nzrvals @ np.log2(nzrvals) - rvals @ P @ np.log2(svals) )/  np.log2(basis)
    # the relative entropy is guaranteed to be >= 0, so we clamp the
    # calculated value to 0 to avoid small violations of the lower bound.
    return max(0, S)


def tr_dist(rho, sigma):
    return np.linalg.norm(rho - sigma, 'nuc') / 2

