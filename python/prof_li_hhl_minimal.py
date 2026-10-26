"""Minimal Prof. Li unstable-shear test: ONE frozen-velocity time slab.

Outputs A*z=b and an HHL-compatible Hermitian embedding H*x=rhs.

Unknowns at each time step j:
  z_j = [Ustar_j (2M); p_j (M); U_j (2M)],  M=Nx*Ny.
A frozen field V=U0 is held fixed for N steps.
No fine reference, repeated slabs, N sweep, or plots.

Python port of prof_li_hhl_minimal.m. Grid vectors are flattened in
column-major (Fortran) order so every operator matches the MATLAB layout.
"""

import os

import numpy as np
import scipy.io
import scipy.sparse as sp
from scipy.sparse.linalg import norm as spnorm
from scipy.sparse.linalg import spsolve

EPS = np.finfo(float).eps
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")


def first_periodic(n, h):
    e = np.ones(n - 1)
    D = sp.diags([-e, e], [-1, 1], shape=(n, n), format="lil") / (2 * h)
    D[0, n - 1] = -1 / (2 * h)
    D[n - 1, 0] = 1 / (2 * h)
    return D.tocsr()


def second_periodic(n, h):
    D2 = sp.diags(
        [np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)],
        [-1, 0, 1],
        shape=(n, n),
        format="lil",
    ) / (h * h)
    D2[0, n - 1] = 1 / (h * h)
    D2[n - 1, 0] = 1 / (h * h)
    return D2.tocsr()


def main():
    # 1. PARAMETERS
    Nx, Ny = 5, 5      # HHL toy test. For original grid use 21, 21.
    N = 2              # Frozen steps in ONE slab (start with 1 or 2).
    alpha = 0.70
    rho = 1.0
    nu = 0.01
    epsilon = 0.01
    dt = 0.005
    Lx = 2 * np.pi / alpha
    Ly = 2 * np.pi

    assert Nx % 2 == 1 and Ny % 2 == 1, (
        "Use odd grid counts: centered periodic D*G has extra null modes on even grids."
    )
    M = Nx * Ny
    q = 5 * M          # Unknowns per time step
    K = q * N          # Unknowns over N steps

    # 2. PERIODIC FINITE-DIFFERENCE OPERATORS
    hx = Lx / Nx
    hy = Ly / Ny
    D1x = first_periodic(Nx, hx)
    D1y = first_periodic(Ny, hy)
    D2x = second_periodic(Nx, hx)
    D2y = second_periodic(Ny, hy)
    Dx = sp.kron(sp.eye(Ny), D1x, format="csr")
    Dy = sp.kron(D1y, sp.eye(Nx), format="csr")
    L = (sp.kron(sp.eye(Ny), D2x) + sp.kron(D2y, sp.eye(Nx))).tocsr()
    D = sp.hstack([Dx, Dy], format="csr")   # Divergence: M x 2M
    G = sp.vstack([Dx, Dy], format="csr")   # Gradient: 2M x M

    # Pressure equation uses exactly Lp=D*G (not the 5-point Laplacian L).
    Ap = (D @ G).tolil()
    Ap[0, :] = 0                            # Pressure gauge: p at first node = 0
    Ap[0, 0] = 1
    Ap = Ap.tocsr()
    Dg = D.tolil()
    Dg[0, :] = 0                            # Consistent gauge row on pressure RHS
    Dg = Dg.tocsr()

    # 3. UNSTABLE-SHEAR INITIAL VELOCITY AND BODY FORCE
    x = np.arange(Nx) * hx
    y = np.arange(Ny) * hy
    X, Y = np.meshgrid(x, y, indexing="ij")
    Xv = X.ravel(order="F")
    Yv = Y.ravel(order="F")
    uBase = -np.sin(Yv)
    Ubase = np.concatenate([uBase, np.zeros(M)])
    U0 = np.concatenate([uBase, (epsilon / alpha) * np.sin(alpha * Xv)])

    # Discrete body force balances viscosity for steady base flow.
    F = -nu * (sp.block_diag([L, L]) @ Ubase)
    F[:M] -= F[:M].mean()
    F[M:] -= F[M:].mean()

    # 4. FREEZE CONVECTION AND FORM ONE-STEP PROPAGATOR
    Vx = U0[:M]
    Vy = U0[M:]
    C = sp.diags(Vx) @ Dx + sp.diags(Vy) @ Dy
    S = sp.eye(M) - dt * C + nu * dt * L
    B = sp.block_diag([S, S], format="csr")  # Ustar_j = B*U_(j-1) + dt*F

    # 5. ASSEMBLE THE GLOBAL NON-HERMITIAN LINEAR SYSTEM A*z=b
    # At each step:
    #   Ustar_j - B*U_(j-1)             = dt*F
    #   -(rho/dt)*Dg*Ustar_j + Ap*p_j  = 0
    #   -Ustar_j + (dt/rho)*G*p_j + U_j = 0
    I2 = sp.eye(2 * M)
    A0 = sp.bmat([
        [I2,                 None,             None],
        [-(rho / dt) * Dg,   Ap,               None],
        [-I2,                (dt / rho) * G,   I2],
    ])

    # Off-diagonal coupling: previous projected velocity U_(j-1).
    A1 = sp.bmat([
        [None, None, -B],
        [None, sp.csr_matrix((M, M)), None],
        [sp.csr_matrix((2 * M, 2 * M)), None, None],
    ])
    shift = sp.eye(N, k=-1)
    A = (sp.kron(sp.eye(N), A0) + sp.kron(shift, A1)).tocsc()
    A.eliminate_zeros()

    b = np.zeros(K)
    for j in range(N):
        b[j * q : j * q + 2 * M] = dt * F
    b[: 2 * M] += B @ U0

    # 6. CLASSICAL VALIDATION (NOT AN HHL COMPUTATION)
    z_ref = spsolve(A, b)
    U = U0.copy()
    for _ in range(N):
        Ustar = B @ U + dt * F
        p = spsolve(Ap.tocsc(), (rho / dt) * (Dg @ Ustar))
        U = Ustar - (dt / rho) * (G @ p)
    U_from_A = z_ref[(N - 1) * q + 3 * M : (N - 1) * q + 5 * M]
    rel_error = np.linalg.norm(U_from_A - U) / max(np.linalg.norm(U), EPS)
    residual = np.linalg.norm(A @ z_ref - b) / max(np.linalg.norm(b), EPS)
    print(f"Grid {Nx}x{Ny}; N={N}; A={K}x{K}; nnz(A)={A.nnz}")
    print(f"Relative difference from sequential solver: {rel_error:.3e}")
    print(f"Relative linear-system residual: {residual:.3e}")

    # 7. HHL INPUT: HERMITIAN EMBEDDING
    # H0*[0;z] = [b;0] with H0=[0,A;A',0].
    # Scaling both sides by the same factor does not change the solution.
    H0 = sp.bmat([[None, A], [A.conj().T, None]], format="csc")
    scale = spnorm(H0, 1)                   # Upper bound for spectral norm
    H = H0 / scale
    rhs = np.concatenate([b, np.zeros(K)]) / scale

    # H is Hermitian but can be poorly conditioned; no HHL is run here.
    hermitian_residual = (
        np.linalg.norm(H @ np.concatenate([np.zeros(K), z_ref]) - rhs)
        / max(np.linalg.norm(rhs), EPS)
    )
    print(f"HHL Hermitian matrix: {2 * K}x{2 * K}; nnz(H)={H.nnz}")
    print(f"Hermitian embedding residual: {hermitian_residual:.3e}")

    # 8. EXPORT to ../output (MATLAB -v5 format; open with load() or scipy.io.loadmat)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outFile = os.path.join(OUTPUT_DIR, "prof_li_hhl_case.mat")
    scipy.io.savemat(outFile, {
        "A": A, "b": b, "H": H, "rhs": rhs, "z_ref": z_ref, "U0": U0, "U": U,
        "Nx": Nx, "Ny": Ny, "N": N, "M": M, "dt": dt, "rho": rho, "nu": nu,
        "alpha": alpha, "epsilon": epsilon, "scale": scale,
    }, oned_as="column")
    print(f"Saved {os.path.normpath(outFile)}")


if __name__ == "__main__":
    main()
