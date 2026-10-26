"""Two-stage reduction of Prof. Li's frozen-velocity Navier-Stokes matrix,
rebuilding the matrix after each HHL solve.

  1. U_j = Z*a_j, where D*Z=0 and Z'*Z=I.
  2. Ared*a = bred, Ared = I - kron(S_N,K), K=Z'*B(V)*Z.
  3. P5 = I+X+X^2+X^3+X^4, X=kron(S_N,K).
     Solve A5*a=b5, A5=I-X^5, b5=P5*bred.
  4. Recover all physical velocities, pressures and intermediate
     velocities; use LAST velocity to rebuild B(V), K, A5 next slab.

No quantum library is called. Classical A5\\b5 is a VERIFIED placeholder
for the student's HHL platform. Exported .mat files are readable by
MATLAB load() and scipy.io.loadmat.

IMPORTANT: HHL outputs a NORMALIZED solution state. To propagate between
slabs, the student must recover a correctly scaled classical velocity
vector. A state vector with unknown amplitude is insufficient.

Python port of prof_li_hhl_reduced_multislab.m. Grid vectors are flattened
in column-major (Fortran) order so every operator matches the MATLAB layout.
Z comes from an SVD null space, as MATLAB's null() does, but the particular
orthonormal basis (and hence a, K, A5) can differ from MATLAB's by a rotation;
U_j, p_j and the condition numbers do not depend on that choice.
"""

import os
import warnings

import numpy as np
import scipy.io
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.linalg import norm as spnorm
from scipy.sparse.linalg import matrix_power as sp_matrix_power
from scipy.sparse.linalg import spsolve, splu

EPS = np.finfo(float).eps
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")


def periodic_first(n, h):
    e = np.ones(n - 1)
    D = sp.diags([-e, e], [-1, 1], shape=(n, n), format="lil") / (2 * h)
    D[0, n - 1] = -1 / (2 * h)
    D[n - 1, 0] = 1 / (2 * h)
    return D.tocsr()


def periodic_second(n, h):
    D2 = sp.diags(
        [np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)],
        [-1, 0, 1],
        shape=(n, n),
        format="lil",
    ) / (h * h)
    D2[0, n - 1] = 1 / (h * h)
    D2[n - 1, 0] = 1 / (h * h)
    return D2.tocsr()


def student_hhl_adapter(H, rhsH, A5, b5, a_ref, slab):
    """TO BE REPLACED BY THE STUDENT:
    1. Send H, rhsH to HHL (Hermitian system).
    2. HHL should return the direction of [0;a], NOT [a;0].
    3. Recover its physical amplitude and global phase/sign so that
       A5*a approximately equals b5. A normalized state is NOT enough.
    4. Return a (length A5.shape[0]) in the same ordering as a_classical.
    For classical prototyping, an approximate direction a_unit may be
    rescaled with t = ((A5@a_unit) . b5) / ||A5@a_unit||^2.
    This uses a FULL classical vector and is only a verification technique,
    NOT an end-to-end quantum advantage.
    """
    raise NotImplementedError(f"Implement student_hhl_adapter for slab {slab}.")


def build_original_full(B, f, Ap, Dg, G, rho, dt, M, n, U0):
    """Exactly matches the original full per-step unknowns [Ustar;p;U]."""
    q = 5 * M
    I2 = sp.eye(2 * M)
    A0 = sp.bmat([
        [I2,                 None,             None],
        [-(rho / dt) * Dg,   Ap,               None],
        [-I2,                (dt / rho) * G,   I2],
    ])
    A1 = sp.bmat([
        [None, None, -B],
        [None, sp.csr_matrix((M, M)), None],
        [sp.csr_matrix((2 * M, 2 * M)), None, None],
    ])
    S = sp.eye(n, k=-1)
    A = (sp.kron(sp.eye(n), A0) + sp.kron(S, A1)).tocsc()
    b = np.zeros(q * n)
    for j in range(n):
        b[j * q : j * q + 2 * M] = f
    b[: 2 * M] += B @ U0
    return A, b


def main():
    # ----------------------- USER SETTINGS -------------------------------
    Nx, Ny = 5, 5           # Initial HHL test. Change both to 21 later.
    Nfreeze = 10            # Frozen steps per solve (>=6 for nontrivial A5).
    totalSteps = 20         # Two slabs: solve, reconstruct, REBUILD, solve.
    alpha = 0.70
    rho = 1.0
    nu = 0.01
    epsilon = 0.01
    dt = 0.005
    powerK = 5              # P5; keep as 5 for our previous transformation.
    useHHLadapter = False   # Change to True after implementing stub above.
    exportMatrices = True   # One .mat file per slab, for Python/HHL.
    verifyFullMatrix = True # Full original system check ONLY on small grids.
    outputDir = os.path.normpath(os.path.join(OUTPUT_DIR, "hhl_reduced_cases"))

    assert Nx % 2 == 1 and Ny % 2 == 1, (
        "Use odd Nx,Ny: centered periodic D*G has extra null modes for even grids."
    )
    assert Nfreeze >= 1 and totalSteps >= 1 and powerK >= 1
    if Nfreeze <= powerK:
        warnings.warn(
            f"With Nfreeze <= {powerK}, X^{powerK}=0 and A{powerK}=I; "
            f"this is a TRIVIAL HHL test. Use Nfreeze>{powerK}."
        )
    # summary.mat is always written, so the folder is always needed.
    os.makedirs(outputDir, exist_ok=True)
    M = Nx * Ny
    q = 5 * M
    Lx = 2 * np.pi / alpha
    Ly = 2 * np.pi
    dx = Lx / Nx
    dy = Ly / Ny

    # --------------------- SPACE OPERATORS ------------------------------
    Dx1 = periodic_first(Nx, dx)
    Dy1 = periodic_first(Ny, dy)
    Dxx1 = periodic_second(Nx, dx)
    Dyy1 = periodic_second(Ny, dy)
    Dx = sp.kron(sp.eye(Ny), Dx1, format="csr")
    Dy = sp.kron(Dy1, sp.eye(Nx), format="csr")
    L = (sp.kron(sp.eye(Ny), Dxx1) + sp.kron(Dyy1, sp.eye(Nx))).tocsr()
    D = sp.hstack([Dx, Dy], format="csr")   # M-by-2M: divergence
    G = sp.vstack([Dx, Dy], format="csr")   # 2M-by-M: pressure gradient
    Ap = (D @ G).tolil()                    # PRESSURE Laplacian, not 5-point L
    Ap[0, :] = 0
    Ap[0, 0] = 1
    Ap = Ap.tocsc()
    Dg = D.tolil()
    Dg[0, :] = 0                            # Must be modified in SAME row as Ap
    Dg = Dg.tocsr()
    Ap_lu = splu(Ap)                        # Ap is unchanged across slabs

    # Compute divergence-free orthonormal basis ONCE (geometry is unchanged).
    # A dense SVD null space is deliberately simple; suitable for small
    # demonstration grids, but scales poorly. It is NOT an efficient
    # quantum encoding.
    Z = scipy.linalg.null_space(D.toarray())
    r = Z.shape[1]
    assert np.linalg.norm(D @ Z) / max(spnorm(D), EPS) < 1e-10, (
        "Z fails divergence-free check."
    )
    assert np.linalg.norm(Z.T @ Z - np.eye(r)) / max(np.sqrt(r), EPS) < 1e-10, (
        "Z is not orthonormal."
    )
    print(f"Grid {Nx} x {Ny}: M={M}, rank(D)={2 * M - r}, reduced velocity DOF={r}")

    # -------------- UNSTABLE SHEAR + DISCRETE BODY FORCE ----------------
    xg, yg = np.meshgrid(np.arange(Nx) * dx, np.arange(Ny) * dy, indexing="ij")
    xv = xg.ravel(order="F")
    yv = yg.ravel(order="F")
    uBase = -np.sin(yv)
    Ubase = np.concatenate([uBase, np.zeros(M)])
    U0 = np.concatenate([uBase, (epsilon / alpha) * np.sin(alpha * xv)])
    F = -nu * (sp.block_diag([L, L]) @ Ubase)
    F[:M] -= F[:M].mean()
    F[M:] -= F[M:].mean()
    f = dt * F                              # Known constant contribution to Ustar

    # Initial field must lie in ker(D); reconstruct to remove roundoff.
    a_in = Z.T @ U0
    U_in = Z @ a_in
    assert np.linalg.norm(U_in - U0) / max(np.linalg.norm(U0), EPS) < 1e-9, (
        "Initial velocity is not discretely divergence-free."
    )

    Uhistory = np.zeros((2 * M, totalSteps + 1))
    Uhistory[:, 0] = U_in
    Phistory = np.zeros((M, totalSteps))
    residualReduced = []
    residualFull = []
    recoveryError = []
    kappaRed = []
    kappa5 = []

    # ------------------ ONE HHL SOLVE PER FROZEN SLAB --------------------
    nDone = 0
    slab = 0
    while nDone < totalSteps:
        slab += 1
        n = min(Nfreeze, totalSteps - nDone)  # Steps in this slab
        V = U_in                              # Freeze THIS velocity for n steps

        # REBUILD the convection operator and propagator each slab.
        C = sp.diags(V[:M]) @ Dx + sp.diags(V[M:]) @ Dy
        Svel = sp.eye(M) - dt * C + nu * dt * L
        B = sp.block_diag([Svel, Svel], format="csr")

        # Spatial reduction: U=Z*a, so a_j=K*a_(j-1)+g.
        K = Z.T @ (B @ Z)
        g = Z.T @ f
        bred = np.tile(g, n)
        bred[:r] = K @ a_in + g

        # Ared = I-X, X = S_n (subdiagonal shift) kron K.
        S_n = sp.eye(n, k=-1, format="csr")
        X = sp.kron(S_n, sp.csr_matrix(K), format="csr")
        nr = n * r
        Ared = (sp.eye(nr) - X).tocsc()

        # LEFT transformation P5:
        # (I+X+...+X^4) (I-X) = I-X^5.
        # Transform RHS too. Construct without dense P5.
        Kp = np.linalg.matrix_power(K, powerK)
        Sp = sp_matrix_power(S_n, powerK)
        A5 = (sp.eye(nr) - sp.kron(Sp, sp.csr_matrix(Kp))).tocsc()
        b5 = bred.copy()
        term = bred
        for _ in range(powerK - 1):
            term = X @ term
            b5 = b5 + term

        # Hermitian embedding for HHL: H*[0;a]=[b5;0].
        H0 = sp.bmat([[None, A5], [A5.conj().T, None]], format="csc")
        scale = spnorm(H0, 1)               # Bound ||H0||_2, for spectral normalization.
        H = H0 / scale
        rhsH = np.concatenate([b5, np.zeros(nr)]) / scale

        # This is the REFERENCE for checking the HHL solution.
        a_classical = spsolve(A5, b5)

        if exportMatrices:
            filename = os.path.join(outputDir, f"slab_{slab:03d}.mat")
            scipy.io.savemat(filename, {
                "Nx": Nx, "Ny": Ny, "M": M, "n": n, "r": r, "Nfreeze": Nfreeze,
                "powerK": powerK, "dt": dt, "rho": rho, "nu": nu,
                "epsilon": epsilon, "alpha": alpha, "V": V, "a_in": a_in,
                "Z": Z, "K": K, "g": g, "Ared": Ared, "bred": bred, "A5": A5,
                "b5": b5, "H": H, "rhsH": rhsH, "scale": scale,
                "a_classical": a_classical,
            }, oned_as="column", do_compression=True)

        # ----- Replace this ONE solve with student HHL output -----
        if useHHLadapter:
            a = student_hhl_adapter(H, rhsH, A5, b5, a_classical, slab)
        else:
            a = a_classical
        # ----------------------------------------------------------

        a = np.asarray(a).ravel()
        assert a.size == nr, "HHL solver must return n*r reconstructed amplitudes."
        rred = np.linalg.norm(Ared @ a - bred) / max(np.linalg.norm(bred), EPS)
        r5 = np.linalg.norm(A5 @ a - b5) / max(np.linalg.norm(b5), EPS)
        errA = np.linalg.norm(a - a_classical) / max(np.linalg.norm(a_classical), EPS)
        residualReduced.append(r5)

        # Recover every physical state from a_j, and the full [Ustar;q;U].
        # This recovery is also a physical consistency test (pressure gauge).
        full_z = np.zeros(q * n)
        Uprev = U_in
        errRecover = 0.0
        for j in range(n):
            Uj = Z @ a[j * r : (j + 1) * r]   # Corrected velocity U_j
            Ustar = B @ Uprev + f             # Intermediate velocity
            qj = Ap_lu.solve(Dg @ Ustar)      # q_j = (dt/rho)*p_j
            pj = (rho / dt) * qj              # Physical pressure
            Uproject = Ustar - G @ qj         # Independent Chorin reconstruction

            errRecover = max(
                errRecover,
                np.linalg.norm(Uj - Uproject) / max(np.linalg.norm(Uproject), EPS),
            )
            full_z[j * q : (j + 1) * q] = np.concatenate([Ustar, pj, Uj])

            Uhistory[:, nDone + j + 1] = Uj
            Phistory[:, nDone + j] = pj
            Uprev = Uj
        recoveryError.append(errRecover)

        # Optional ORIGINAL 5M-by-5M block matrix check, FIRST SLAB only.
        fullResidual = np.nan
        fullDiff = np.nan
        if verifyFullMatrix and slab == 1 and q * n <= 1500:
            Afull, bfull = build_original_full(B, f, Ap, Dg, G, rho, dt, M, n, U_in)
            fullResidual = (
                np.linalg.norm(Afull @ full_z - bfull) / max(np.linalg.norm(bfull), EPS)
            )
            zfull_ref = spsolve(Afull, bfull)
            fullDiff = (
                np.linalg.norm(zfull_ref - full_z) / max(np.linalg.norm(zfull_ref), EPS)
            )
        residualFull.append(fullResidual)

        # Exact 2-norm condition numbers feasible ONLY on small matrices.
        condRed = np.nan
        condA5 = np.nan
        if nr <= 350:
            condRed = np.linalg.cond(Ared.toarray())
            condA5 = np.linalg.cond(A5.toarray())
        kappaRed.append(condRed)
        kappa5.append(condA5)

        print(f"Slab {slab}: steps {nDone + 1}-{nDone + n} | full {q * n}, reduced {nr} | "
              f"kappa(Ared)={condRed:.4g}, kappa(A5)={condA5:.4g}")
        print(f"  transformed residual {r5:.3e}, Ared residual {rred:.3e}, "
              f"HHL/classical difference {errA:.3e}")
        print(f"  recovered projection mismatch {errRecover:.3e}; "
              f"full residual {fullResidual:.3e}; full solution mismatch {fullDiff:.3e}")

        # KEY REBUILD STEP: updated physical U starts the NEXT frozen block.
        U_in = Uprev
        a_in = Z.T @ U_in
        nDone += n

    t = np.arange(totalSteps + 1) * dt
    scipy.io.savemat(os.path.join(outputDir, "summary.mat"), {
        "Uhistory": Uhistory, "Phistory": Phistory, "t": t,
        "residualReduced": np.array(residualReduced),
        "residualFull": np.array(residualFull),
        "recoveryError": np.array(recoveryError),
        "kappaRed": np.array(kappaRed), "kappa5": np.array(kappa5),
        "Nx": Nx, "Ny": Ny, "Nfreeze": Nfreeze, "totalSteps": totalSteps,
    }, oned_as="column", do_compression=True)
    print("DONE. Updated B(V), K, Ared, A5 after every solved slab.")
    print(f"All physical states and slab HHL matrices exported to {outputDir}")


if __name__ == "__main__":
    main()
