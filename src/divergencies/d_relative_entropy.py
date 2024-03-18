import numpy as np


def d_relative_entropy(rho, sigma):
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
    S = nzrvals @ np.log2(nzrvals) - rvals @ P @ np.log2(svals)
    # the relative entropy is guaranteed to be >= 0, so we clamp the
    # calculated value to 0 to avoid small violations of the lower bound.
    return max(0, S)


if __name__ == "__main__":
    import qutip
    dim = 3

    rho = np.zeros((dim, dim))
    rho[0,0] = 1
    sigma = np.zeros((dim, dim))
    sigma[1,1] = 1

    print(d_relative_entropy(rho, sigma))

    print(qutip.entropy_relative(qutip.Qobj(rho), qutip.Qobj(sigma), base=2))