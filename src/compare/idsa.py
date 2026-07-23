"""iDSA: input-driven dynamics (InputDSA).  [iDSA member track; plan 2.4/2.5/2.6]

Ready and Set are strong external drive, so plain DSA (which looks at intrinsic
dynamics only) is not enough here. InputDSA separates the input-driven part from
the recurrent part before comparing (AGENTS.md, "iDSA, not plain DSA"). The recipe:
fit DMDc-based input and recurrent operators on each system's trajectories together
with their stored inputs, then compare the two systems.

Stages, built in order (see plan decision 11):
    Stage 3  BPTT-vs-PC        rule-vs-rule dynamics distance   answers RQ1  (2.4)
    Stage 4  model-to-DMFC     each model vs DMFC               answers RQ3  (2.5)
    2.6      across-ts         Stage 3/4 resolved by interval   answers RQ2

Both systems have to meet the same requirements: the same conditions, inputs
aligned to states, identical preprocessing, matched k and time bins, and finite,
reproducible distances. Distances come back per seed.

Method (Huang, Ostrow, Singh, Kozachkov, Fiete, Rajan 2025, arXiv 2510.25943).
For each system, fit a linear input-driven model  x_{t+1} = A x_t + B u_t  from the
trajectories and their aligned inputs, then compare the two systems on those
operators. There are two estimators (paper Sec 2.3, Alg 1/3):
      * ``dmdc``     Dynamic Mode Decomposition with control (Alg 3). Correct and
                     cheap when the state is fully observed, which our model latents
                     are once they sit in a shared PCA space. Default for Stage 3.
      * ``subspace`` Subspace DMDc / N4SID on lifted states (Alg 1). Needed under
                     partial observation, i.e. neural data where you record a
                     handful of units from a large population and plain DMDc biases
                     B toward the intrinsic dynamics (paper Sec 2.3, "Issues of
                     Partial Observation"). Use it for Stage 4 model-to-DMFC.
The distance (paper Eq. 8/9/10, Appendix G): line the two systems up with a single
orthogonal C, found in closed form by Procrustes on the controllability matrix
K = [B, AB, A^T B, A^2 B, ...], then read off
      * controllability distance ||C K1 - K2||_F   (the InputDSA scalar, Eq 8)
      * state distance           ||C A1 C^T - A2||_F  (intrinsic dynamics, Eq 9)
      * input distance           ||C B1 - B2||_F   (input drive, Eq 10)
This is what iDSA buys over plain DSA: two systems that share recurrent dynamics
but have different input matrices stay close in state distance even when their
input distance is large.

Two backends, one interface. By default this calls the official InputDSA package
(``dsa-metric``, the mitchellostrow/DSA repo) to fit the operators and compute the
distance, selected by ``InputDSAConfig.backend='dsa-metric'``. That package pulls
torch and pydmd and wants numpy 2.5, so it lives in its own env (requirements-idsa.txt)
apart from the modeling env, per AGENTS.md's dependency rule. When it is not
importable the code falls back to ``backend='builtin'``: a numpy/scipy-only
reimplementation from the paper that runs anywhere and doubles as a cross-check (its
distances agree with the official package to about 1e-2 on synthetic systems). Both
backends share the same ``fit_operators`` / ``input_dsa`` / ``Operators`` interface,
so callers never change. The builtin estimators (``dmdc`` / ``subspace_dmdc``) and the
controllability-Procrustes metric below are that fallback.

Definition of done:
    Stage 3: finite, reproducible BPTT/PC distances on smoke-scale data.
    Stage 4: per-model, per-seed distance to DMFC with the neural noise ceiling.

Reference: DSA/InputDSA at https://github.com/mitchellostrow/DSA, and the InputDSA
paper at https://arxiv.org/abs/2510.25943.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration and the fitted-operator container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputDSAConfig:
    """Knobs for operator estimation and the similarity metric.

    Keep these identical across the two systems in any one comparison. A mismatch in
    rank or input dimensionality makes the operators incommensurable (AGENTS.md,
    "matched k and time bins").
    """

    backend: str = "dsa-metric"   # "dsa-metric" (official pkg) or "builtin" (numpy fallback)
    method: str = "dmdc"          # "dmdc" (Alg 3) or "subspace" (Alg 1, partial obs)
    rank: int = 10                # state-space rank r; matched across systems
    delays: int = 1              # delay-embedding order q (1 = no lift). Subspace uses >1
    ridge: float = 1e-6           # Tikhonov regularization on the least-squares fits
    n_powers: int = 8             # # of controllability powers in K (paper caps for stability)
    alpha: float = 0.5            # state<->input weighting in the joint scalar (Eq 8/H)
    augment_transpose: bool = True  # add A^T powers to K for a better-conditioned C (App. G)
    power_norm_cap: float = 1e6   # stop adding powers once a block's norm blows up


@dataclass
class Operators:
    """A fitted input-driven linear model of one system.

    ``A`` is the r x r state-transition (recurrent/intrinsic) operator, ``B`` the
    r x n_in input-to-state operator. ``readout`` (Subspace DMDc only) is the
    observation matrix C from the extended observability matrix; kept for
    diagnostics, not used by the distance.
    """

    A: np.ndarray                 # [r, r]
    B: np.ndarray                 # [r, n_in]
    rank: int
    method: str
    delays: int
    readout: Optional[np.ndarray] = None      # [n_obs, r], Subspace DMDc
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A must be square [r, r]; got {self.A.shape}")
        if self.B.ndim != 2 or self.B.shape[0] != self.A.shape[0]:
            raise ValueError(
                f"B must be [r, n_in] with r matching A; got A{self.A.shape} B{self.B.shape}"
            )


# ---------------------------------------------------------------------------
# Snapshot construction (shared by both estimators)
# ---------------------------------------------------------------------------


def _as_trajectories(states: np.ndarray, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce to [n_traj, time, dim] and validate that states and inputs align in time.

    Accepts a single trajectory [time, dim] or a batch [n_traj, time, dim]. Each
    condition (or trial) is one trajectory; the DMD is fit over all of them jointly.
    """
    states = np.asarray(states, dtype=float)
    inputs = np.asarray(inputs, dtype=float)
    if states.ndim == 2:
        states = states[None]
    if inputs.ndim == 2:
        inputs = inputs[None]
    if states.ndim != 3 or inputs.ndim != 3:
        raise ValueError(
            "states/inputs must be [time, dim] or [n_traj, time, dim]; "
            f"got states{states.shape} inputs{inputs.shape}"
        )
    if states.shape[:2] != inputs.shape[:2]:
        raise ValueError(
            "states and inputs must share [n_traj, time] (inputs aligned to states); "
            f"got states{states.shape[:2]} inputs{inputs.shape[:2]}"
        )
    return states, inputs


def _delay_embed(traj: np.ndarray, q: int) -> np.ndarray:
    """Hankel delay embedding of one [time, dim] trajectory -> [time-q+1, dim*q].

    Row t stacks [x_t, x_{t-1}, ..., x_{t-q+1}] (paper Sec 2.3: phi is a delay
    embedding). q == 1 returns the trajectory unchanged.
    """
    if q <= 1:
        return traj
    T = traj.shape[0]
    if T < q:
        raise ValueError(f"trajectory of length {T} too short for {q} delays")
    return np.concatenate([traj[q - 1 - i: T - i] for i in range(q)], axis=1)


def _snapshot_pairs(
    states: np.ndarray, inputs: np.ndarray, q: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build DMDc snapshot columns from all trajectories, respecting boundaries.

    Returns ``X_prev`` [feat, N], ``X_next`` [feat, N], ``U_prev`` [n_in, N]. The
    columns come from consecutive pairs within a single trajectory only. A pair is
    never formed across a condition boundary, since that jump is not real dynamics.
    """
    states, inputs = _as_trajectories(states, inputs)
    x_prev, x_next, u_prev = [], [], []
    for s, u in zip(states, inputs):
        se = _delay_embed(s, q)          # [T', dim*q]
        # inputs align to the *current* (embedded) state's newest time index
        ue = u[q - 1:] if q > 1 else u   # [T', n_in]
        if se.shape[0] < 2:
            continue
        x_prev.append(se[:-1].T)         # [feat, T'-1]
        x_next.append(se[1:].T)
        u_prev.append(ue[:-1].T)         # [n_in, T'-1]
    if not x_prev:
        raise ValueError("no usable snapshot pairs (trajectories too short)")
    return (
        np.concatenate(x_prev, axis=1),
        np.concatenate(x_next, axis=1),
        np.concatenate(u_prev, axis=1),
    )


# ---------------------------------------------------------------------------
# Estimator 1: DMDc (paper Algorithm 3), for fully observed model latents
# ---------------------------------------------------------------------------


def dmdc(states: np.ndarray, inputs: np.ndarray, cfg: Optional[InputDSAConfig] = None) -> Operators:
    """Fit A, B by Dynamic Mode Decomposition with control (paper Alg 3).

    Solves [A B] = X_next [X_prev; U_prev]^+ with ridge, then projects the state
    operator into the leading rank-r POD subspace of X_next so A is r x r (the
    comparison space). Correct when the state is fully observed; for partial
    observation use :func:`subspace_dmdc`.
    """
    cfg = cfg or InputDSAConfig()
    x_prev, x_next, u_prev = _snapshot_pairs(states, inputs, cfg.delays)
    n_in = u_prev.shape[0]

    # POD basis of the (embedded) state: the shared r-dim comparison coordinates.
    Ur, _, _ = np.linalg.svd(x_next, full_matrices=False)
    r = min(cfg.rank, Ur.shape[1])
    Ur = Ur[:, :r]

    # Ridge least squares for [A_full | B_full] against stacked [state; input].
    omega = np.vstack([x_prev, u_prev])                      # [feat+n_in, N]
    g = omega @ omega.T
    g[: g.shape[0]] += cfg.ridge * np.eye(g.shape[0])
    coef = (x_next @ omega.T) @ np.linalg.pinv(g)            # [feat, feat+n_in]
    a_full = coef[:, : x_prev.shape[0]]
    b_full = coef[:, x_prev.shape[0]:]

    # Project into the rank-r state subspace (paper Alg 3, COMPUTEOPERATORS step).
    A = Ur.T @ a_full @ Ur                                   # [r, r]
    B = Ur.T @ b_full                                        # [r, n_in]
    return Operators(
        A=A, B=B, rank=r, method="dmdc", delays=cfg.delays,
        meta={"n_in": n_in, "n_snapshots": x_prev.shape[1], "basis": Ur},
    )


# ---------------------------------------------------------------------------
# Estimator 2: Subspace DMDc / N4SID on lifted states (paper Algorithm 1)
# ---------------------------------------------------------------------------


def _block_hankel(mat: np.ndarray, window: int, start: int, T: int) -> np.ndarray:
    """Block-Hankel of a [dim, N] signal: rows stacked over ``window`` shifts.

    Column t (0..T-1) is [mat[:, start+t], mat[:, start+t+1], ..., stacked].
    """
    dim = mat.shape[0]
    out = np.empty((dim * window, T), dtype=float)
    for i in range(window):
        out[i * dim:(i + 1) * dim] = mat[:, start + i: start + i + T]
    return out


def subspace_dmdc(
    states: np.ndarray, inputs: np.ndarray, cfg: Optional[InputDSAConfig] = None
) -> Operators:
    """Fit A, B, C by Subspace DMDc, which is N4SID run on lifted states (paper Alg 1).

    This one holds up under partial observation and under observation or process
    noise. It estimates a latent state sequence by oblique projection, which removes
    the direct input pathway that otherwise biases plain DMDc's B, then regresses the
    shifted states. Use it for neural data (Stage 4). Each condition's trajectory is
    processed on its own and the projections are concatenated, so boundaries stay
    intact.
    """
    cfg = cfg or InputDSAConfig()
    states, inputs = _as_trajectories(states, inputs)
    p = f = max(cfg.delays, 2)             # past/future windows (need >= 2 for a shift)
    n = cfg.rank

    y_f_blocks, z_p_blocks, u_f_blocks = [], [], []
    for s, u in zip(states, inputs):
        Y = s.T                            # [d, N]
        U = u.T                            # [m, N]
        N = Y.shape[1]
        T = N - p - f + 1
        if T <= n:                         # too short to identify an n-dim state
            continue
        Y_p = _block_hankel(Y, p, 0, T)
        Y_f = _block_hankel(Y, f, p, T)
        U_p = _block_hankel(U, p, 0, T)
        U_f = _block_hankel(U, f, p, T)
        y_f_blocks.append(Y_f)
        u_f_blocks.append(U_f)
        z_p_blocks.append(np.vstack([U_p, Y_p]))     # Z_p = [U_p; Y_p]
    if not y_f_blocks:
        raise ValueError(
            "trajectories too short for Subspace DMDc; reduce delays/rank or use method='dmdc'"
        )
    Y_f = np.concatenate(y_f_blocks, axis=1)
    U_f = np.concatenate(u_f_blocks, axis=1)
    Z_p = np.concatenate(z_p_blocks, axis=1)
    Tt = Y_f.shape[1]

    # Oblique projection: remove the future-input pathway, then project future
    # outputs onto past regressors (paper Alg 1, OBLIQUEPROJECTION).
    G = U_f @ U_f.T + cfg.ridge * np.eye(U_f.shape[0])
    proj_perp = np.eye(Tt) - U_f.T @ np.linalg.solve(G, U_f)
    Y_f_perp = Y_f @ proj_perp
    Z_p_perp = Z_p @ proj_perp
    # Oblique projection of the future onto the past-regressor ROW SPACE: this is
    # the time-indexed O = Gamma * X_hat ([d*f, Tt]). Projecting back through
    # Z_p_perp (via its pseudoinverse) keeps the column/time axis, unlike the bare
    # regressor operator Y_f_perp Z_p_perp^+.
    O = (Y_f_perp @ np.linalg.pinv(Z_p_perp)) @ Z_p_perp   # [d*f, Tt]

    # State sequence and observability matrix from the truncated SVD of O.
    Uo, So, Vot = np.linalg.svd(O, full_matrices=False)
    r = min(n, Uo.shape[1])
    sqrt_s = np.sqrt(So[:r])
    Gamma = Uo[:, :r] * sqrt_s                          # [d*f, r]
    Xhat = (sqrt_s[:, None]) * Vot[:r]                  # [r, T]

    # Regress shifted states on [state; input] for A, B (paper Alg 1, final step).
    # Xhat columns are ordered trajectory-by-trajectory; the shift x_cur->x_nxt must
    # not cross a trajectory boundary, and U_mid must line up with x_cur per block.
    d = states.shape[2]
    x_cur_blocks, x_nxt_blocks, u_mid_blocks = [], [], []
    off = 0
    for s, u in zip(states, inputs):
        T = s.shape[0] - p - f + 1
        if T <= n:
            continue
        block = Xhat[:, off:off + T]
        x_cur_blocks.append(block[:, :-1])
        x_nxt_blocks.append(block[:, 1:])
        u_mid_blocks.append(u.T[:, p: p + T - 1])   # input at the current state's time
        off += T
    x_cur = np.concatenate(x_cur_blocks, axis=1)
    x_nxt = np.concatenate(x_nxt_blocks, axis=1)
    U_mid = np.concatenate(u_mid_blocks, axis=1)

    reg = np.vstack([x_cur, U_mid])
    gram = reg @ reg.T + cfg.ridge * np.eye(reg.shape[0])
    coef = (x_nxt @ reg.T) @ np.linalg.pinv(gram)      # [r, r+m]
    A = coef[:, :r]
    B = coef[:, r:]
    C = Gamma[:d]                                       # first n_obs rows (readout)
    return Operators(
        A=A, B=B, rank=r, method="subspace", delays=cfg.delays, readout=C,
        meta={"n_in": inputs.shape[2], "n_snapshots": Tt},
    )


# ---------------------------------------------------------------------------
# Backend selection: official dsa-metric package, or the builtin numpy fallback
# ---------------------------------------------------------------------------

_DSA_MODULE = None       # cached DSA package handle
_DSA_TRIED = False       # have we attempted the import yet?
_DSA_WARNED = False      # have we already warned about the fallback?


def _get_dsa():
    """Import the official ``DSA`` package once and cache it (None if unavailable).

    The import is lazy on purpose: DSA pulls torch, and forcing that import whenever
    ``idsa`` is imported would slow down (or break) the modeling env, where the
    package is not installed. Only the dsa-metric backend calls this.
    """
    global _DSA_MODULE, _DSA_TRIED
    if not _DSA_TRIED:
        _DSA_TRIED = True
        try:
            import DSA  # official package: pip install dsa-metric (own env)
            _DSA_MODULE = DSA
        except Exception:
            _DSA_MODULE = None
    return _DSA_MODULE


def _resolve_backend(backend: str) -> str:
    """Return the backend to actually use, falling back to builtin when DSA is absent."""
    if backend == "builtin":
        return "builtin"
    if backend == "dsa-metric":
        if _get_dsa() is not None:
            return "dsa-metric"
        global _DSA_WARNED
        if not _DSA_WARNED:
            _DSA_WARNED = True
            warnings.warn(
                "dsa-metric is not importable; using the builtin iDSA backend. Install "
                "it in a separate env (requirements-idsa.txt) to use the official package.",
                RuntimeWarning,
            )
        return "builtin"
    raise ValueError(f"unknown backend {backend!r}; use 'dsa-metric' or 'builtin'")


def _fit_operators_official(states: np.ndarray, inputs: np.ndarray, cfg: InputDSAConfig) -> Operators:
    """Fit A, B with the official DMDc / SubspaceDMDc, wrapped in :class:`Operators`."""
    dsa = _get_dsa()
    states, inputs = _as_trajectories(states, inputs)     # [n_traj, time, dim]
    if cfg.method == "subspace":
        model = dsa.SubspaceDMDc(
            states, control_data=inputs, n_delays=max(cfg.delays, 2),
            rank=cfg.rank, lamb=cfg.ridge, backend="n4sid",
        )
    elif cfg.method == "dmdc":
        model = dsa.DMDc(
            states, control_data=inputs, n_delays=cfg.delays,
            rank_output=cfg.rank, lamb=cfg.ridge,
        )
    else:
        raise ValueError(f"unknown method {cfg.method!r}; use 'dmdc' or 'subspace'")
    model.fit()
    A = np.asarray(model.A, dtype=float)
    B = np.asarray(model.B, dtype=float)
    return Operators(
        A=A, B=B, rank=A.shape[0], method=cfg.method, delays=cfg.delays,
        meta={"backend": "dsa-metric", "n_in": B.shape[1]},
    )


# ---------------------------------------------------------------------------
# Public estimator dispatch
# ---------------------------------------------------------------------------


def fit_operators(
    states: np.ndarray, inputs: np.ndarray, cfg: Optional[InputDSAConfig] = None
) -> Operators:
    """Fit DMDc input + recurrent operators from [cond, time, k] states + aligned inputs.

    ``states``: [n_traj, time, k] (or a single [time, k]); ``inputs``: matching
    [n_traj, time, n_in]. Inputs must be time-aligned to states (AGENTS.md,
    "store/ keeps the input time series"). Returns fitted :class:`Operators`.
    Uses the backend in ``cfg.backend`` (dsa-metric or builtin), and within a backend
    dispatches on ``cfg.method``: ``"dmdc"`` (fully observed) or ``"subspace"``
    (partial observation, i.e. neural data).
    """
    cfg = cfg or InputDSAConfig()
    if _resolve_backend(cfg.backend) == "dsa-metric":
        return _fit_operators_official(states, inputs, cfg)
    if cfg.method == "dmdc":
        return dmdc(states, inputs, cfg)
    if cfg.method == "subspace":
        return subspace_dmdc(states, inputs, cfg)
    raise ValueError(f"unknown method {cfg.method!r}; use 'dmdc' or 'subspace'")


# ---------------------------------------------------------------------------
# Similarity metric (paper Eq. 8/9/10, Appendix G; closed-form Procrustes)
# ---------------------------------------------------------------------------


def controllability_matrix(
    A: np.ndarray, B: np.ndarray, n_powers: int, augment_transpose: bool, norm_cap: float
) -> np.ndarray:
    """K = [B, AB, A^2 B, ...] (paper Eq. 5), optionally augmented with A^T powers.

    The A^T augmentation (Appendix G) improves the conditioning of the orthogonal
    alignment. Power accumulation stops early if a block's norm exceeds ``norm_cap``
    (guards the numerical instability the paper warns about for lambda_max(A) > 1).
    """
    blocks: List[np.ndarray] = [B]
    cur = B
    cur_t = B
    At = A.T
    for _ in range(n_powers):
        cur = A @ cur
        if not np.all(np.isfinite(cur)) or np.linalg.norm(cur) > norm_cap:
            break
        blocks.append(cur)
        if augment_transpose:
            cur_t = At @ cur_t
            if np.all(np.isfinite(cur_t)) and np.linalg.norm(cur_t) <= norm_cap:
                blocks.append(cur_t)
    return np.concatenate(blocks, axis=1)               # [r, n_in * n_blocks]


def _orthogonal_procrustes(K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """Closed-form C minimizing ||C K1 - K2||_F over the orthogonal group O(n).

    Maximizing Tr(C^T K2 K1^T) gives C* = U V^T where K2 K1^T = U S V^T (paper
    Appendix G). The group is O(n), so there is no determinant correction and
    reflections are allowed.
    """
    M = K2 @ K1.T
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def _input_dsa_official(op_a: Operators, op_b: Operators) -> Dict[str, float]:
    """Distance components from the official controllability metric.

    ``ControllabilitySimilarityTransformDist.fit_score(A, B, A_control, B_control)``
    has a confusing signature: ``A``/``B`` are the two systems' STATE matrices and
    ``A_control``/``B_control`` are their INPUT matrices. With
    ``return_distance_components=True`` (euclidean) it returns
    ``(full controllability, state, control)``, matching our dict keys.
    """
    dsa = _get_dsa()
    metric = dsa.ControllabilitySimilarityTransformDist(
        score_method="euclidean", compare="joint", return_distance_components=True,
    )
    controllability, state, control = metric.fit_score(op_a.A, op_b.A, op_a.B, op_b.B)
    return {
        "distance": float(controllability),
        "state_distance": float(state),
        "input_distance": float(control),
    }


def input_dsa(
    op_a: Operators, op_b: Operators, cfg: Optional[InputDSAConfig] = None
) -> Dict[str, float]:
    """Full InputDSA comparison of two fitted systems.

    Returns a dict with:
      * ``distance``: joint controllability distance ||C K1 - K2||_F (Eq. 8). This
        is the scalar most callers want.
      * ``state_distance``: ||C A1 C^T - A2||_F (Eq. 9), the intrinsic/recurrent
        dynamics.
      * ``input_distance``: ||C B1 - B2||_F (Eq. 10), how input is read into state.
    The single orthogonal C is fit once on the controllability matrix and reused, so
    state and input distances are read in a common aligned frame. With
    ``cfg.backend='dsa-metric'`` the official package computes the same three
    quantities instead.
    """
    cfg = cfg or InputDSAConfig()
    if op_a.A.shape != op_b.A.shape:
        raise ValueError(
            f"operators must share rank r for comparison; got {op_a.A.shape} vs {op_b.A.shape}"
        )
    if op_a.B.shape[1] != op_b.B.shape[1]:
        raise ValueError(
            f"operators must share input dim; got {op_a.B.shape[1]} vs {op_b.B.shape[1]}"
        )
    if _resolve_backend(cfg.backend) == "dsa-metric":
        return _input_dsa_official(op_a, op_b)
    K1 = controllability_matrix(
        op_a.A, op_a.B, cfg.n_powers, cfg.augment_transpose, cfg.power_norm_cap
    )
    K2 = controllability_matrix(
        op_b.A, op_b.B, cfg.n_powers, cfg.augment_transpose, cfg.power_norm_cap
    )
    w = min(K1.shape[1], K2.shape[1])            # equal by construction; guard anyway
    K1, K2 = K1[:, :w], K2[:, :w]
    C = _orthogonal_procrustes(K1, K2)

    distance = float(np.linalg.norm(C @ K1 - K2))
    state_distance = float(np.linalg.norm(C @ op_a.A @ C.T - op_b.A))
    input_distance = float(np.linalg.norm(C @ op_a.B - op_b.B))
    return {
        "distance": distance,
        "state_distance": state_distance,
        "input_distance": input_distance,
    }


def dsa_distance(op_a: Operators, op_b: Operators, cfg: Optional[InputDSAConfig] = None) -> float:
    """Scalar dynamical-similarity distance between two fitted operators (Eq. 8).

    Convenience wrapper returning only the joint controllability distance from
    :func:`input_dsa`. Symmetric up to numerical Procrustes error, and finite and
    reproducible for a fixed config (plan 2.4 check).
    """
    return input_dsa(op_a, op_b, cfg)["distance"]


# ---------------------------------------------------------------------------
# Stage orchestration: read the store, apply shared preprocessing, then compare
# ---------------------------------------------------------------------------
#
# These read states and inputs from the activation store and pass both systems
# through the same fitted preprocessor before fitting operators (AGENTS.md,
# "Identical preprocessing"), which keeps raw, unstandardized activity out of the
# comparison. The preprocessor is the Preprocess & RSA track's object
# (src.preprocess.Preprocessor, fitted on a reference). We use its states+inputs
# entry point,
#   preprocessor.transform_with_inputs(states, inputs)
#       -> ([cond, n_time_bins, k], [cond, n_time_bins, n_in])
# which the Preprocess track added for iDSA: states get z-score -> PCA(k) -> warp,
# inputs get the same per-condition warp and nothing else, so the two stay sample-
# aligned for DMDc. That entry point requires all 20 canonical conditions, so we
# always load the full set and subset AFTER preprocessing (never preprocess a band).


def load_system(store, model: str, seed: int, preprocessor) -> Tuple[np.ndarray, np.ndarray]:
    """Read the 20 canonical conditions for (model, seed) and run the shared warp.

    Trajectories are kept ragged (time length varies with ts/tp) and handed to the
    preprocessor as lists, exactly as ``scripts/run_rsa.py::stack_system`` does.
    Returns preprocessed ``(states [20, n_time_bins, k], inputs [20, n_time_bins,
    n_in])`` on a single time base.
    """
    raw_states, raw_inputs = [], []
    for cond in _canonical_conditions():
        rec = store.read(model, seed, cond)
        raw_states.append(np.asarray(rec.states, dtype=float))
        raw_inputs.append(np.asarray(rec.inputs, dtype=float))
    states_pp, inputs_pp = preprocessor.transform_with_inputs(raw_states, raw_inputs)
    return np.asarray(states_pp, dtype=float), np.asarray(inputs_pp, dtype=float)


def _subset(states: np.ndarray, inputs: np.ndarray, conditions) -> Tuple[np.ndarray, np.ndarray]:
    """Select a subset of the 20 preprocessed conditions by their canonical index.

    Subsetting happens on already-preprocessed arrays so every system still passed
    through the identical full-set warp (the preprocessor needs all 20 to fit its
    per-system PCA and time base). Used to resolve a comparison by ts band (2.6).
    """
    from src.conditions import condition_index

    idx = [condition_index(c) for c in conditions]
    return states[idx], inputs[idx]


def stage3_bptt_vs_pc(
    store,
    seeds: Sequence[int],
    preprocessor,
    conditions=None,
    cfg: Optional[InputDSAConfig] = None,
    model_a: str = "bptt",
    model_b: str = "pc",
) -> Dict[int, Dict[str, float]]:
    """Stage 3 (plan 2.4, RQ1): per-seed BPTT-vs-PC dynamics distance.

    For each seed, fit operators for both learning rules on identically preprocessed
    latents and return the InputDSA distances. Seeds are the unit of evidence
    (AGENTS.md), so this returns the per-seed spread rather than a point estimate.
    ``conditions`` optionally restricts the comparison to a subset (e.g. a ts band),
    applied after preprocessing.
    """
    cfg = cfg or InputDSAConfig()
    out: Dict[int, Dict[str, float]] = {}
    for seed in seeds:
        s_a, u_a = load_system(store, model_a, seed, preprocessor)
        s_b, u_b = load_system(store, model_b, seed, preprocessor)
        if conditions is not None:
            s_a, u_a = _subset(s_a, u_a, conditions)
            s_b, u_b = _subset(s_b, u_b, conditions)
        op_a = fit_operators(s_a, u_a, cfg)
        op_b = fit_operators(s_b, u_b, cfg)
        out[seed] = input_dsa(op_a, op_b, cfg)
    return out


def stage4_model_to_dmfc(
    store,
    models: Sequence[str],
    seeds: Sequence[int],
    preprocessor,
    dmfc_states: np.ndarray,
    dmfc_inputs: np.ndarray,
    conditions=None,
    cfg: Optional[InputDSAConfig] = None,
) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Stage 4 (plan 2.5, answers RQ3): each trained model against DMFC. Main deliverable.

    ``dmfc_states``/``dmfc_inputs`` are the already-preprocessed neural activity and
    its external-input representation on the shared condition set. Neural data is
    partially observed, so pass ``cfg.method='subspace'``. Returns per (model, seed)
    distances to DMFC. Schedule neural ingestion early so this stage isn't left
    stranded at the end (AGENTS.md; plan 2.5).
    """
    cfg = cfg or InputDSAConfig(method="subspace")
    dmfc_states = np.asarray(dmfc_states, dtype=float)
    dmfc_inputs = np.asarray(dmfc_inputs, dtype=float)
    if conditions is not None:
        dmfc_states, dmfc_inputs = _subset(dmfc_states, dmfc_inputs, conditions)
    dmfc_op = fit_operators(dmfc_states, dmfc_inputs, cfg)
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    for model in models:
        for seed in seeds:
            s, u = load_system(store, model, seed, preprocessor)
            if conditions is not None:
                s, u = _subset(s, u, conditions)
            op = fit_operators(s, u, cfg)
            out[(model, seed)] = input_dsa(op, dmfc_op, cfg)
    return out


def across_ts(
    store,
    seeds: Sequence[int],
    preprocessor,
    cfg: Optional[InputDSAConfig] = None,
    dmfc: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    models: Tuple[str, str] = ("bptt", "pc"),
) -> Dict[str, Any]:
    """2.6 (RQ2): the Stage 3/4 contrast resolved by interval band (short vs long).

    Splits the 20 conditions into short- and long-prior bands and runs the comparison
    within each band, giving a signature-vs-ts curve per rule with seed spread (plan
    2.6 check). With ``dmfc=None`` this is Stage 3 (rule-vs-rule) per band. Pass
    ``dmfc=(states, inputs)`` (already preprocessed neural data) for the model-to-DMFC
    contrast per band. Each system is preprocessed on the full 20 conditions, then the
    band is subset out.
    """
    cfg = cfg or InputDSAConfig()
    out: Dict[str, Any] = {}
    for band_name, band_conditions in _ts_bands().items():
        if dmfc is None:
            out[band_name] = stage3_bptt_vs_pc(
                store, seeds, preprocessor, band_conditions, cfg,
                model_a=models[0], model_b=models[1],
            )
        else:
            out[band_name] = stage4_model_to_dmfc(
                store, models, seeds, preprocessor, dmfc[0], dmfc[1],
                conditions=band_conditions, cfg=cfg,
            )
    return out


def _canonical_conditions():
    """The canonical 20 conditions in fixed order (imported lazily to keep the core
    math importable without the src.conditions schema)."""
    from src.conditions import CONDITIONS

    return CONDITIONS


def _ts_bands() -> Dict[str, list]:
    """Group the canonical conditions into short- and long-prior bands (plan 2.6)."""
    bands: Dict[str, list] = {"short": [], "long": []}
    for c in _canonical_conditions():
        bands[c.prior].append(c)
    return bands
