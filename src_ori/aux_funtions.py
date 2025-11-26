import jax.numpy as jnp


def _pchip_slopes(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute PCHIP first-derivative slopes at nodes using the
    Fritsch–Carlson monotone cubic scheme (SciPy-like).

    x, y : (N,) with x strictly increasing.
    Returns m : (N,) slopes at nodes.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    h = jnp.diff(x)              # (N-1,)
    delta = jnp.diff(y) / h      # (N-1,)

    # --- interior slopes m[1..N-2] ---
    # Fritsch–Carlson harmonic mean for same-sign deltas
    h0 = h[:-1]
    h1 = h[1:]
    d0 = delta[:-1]
    d1 = delta[1:]

    w1 = 2.0 * h1 + h0
    w2 = 2.0 * h0 + h1
    m_inner = (w1 + w2) / (w1 / d0 + w2 / d1)  # (N-2,)

    same_sign = (d0 * d1) > 0.0
    m_inner = jnp.where(same_sign, m_inner, 0.0)

    # --- endpoint slopes m[0] and m[-1] (SciPy-style) ---
    # Left endpoint
    m0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    m0 = jnp.where(
        jnp.sign(m0) != jnp.sign(delta[0]),
        0.0,
        m0,
    )
    m0 = jnp.where(
        (jnp.sign(delta[0]) != jnp.sign(delta[1])) & (jnp.abs(m0) > jnp.abs(3.0 * delta[0])),
        0.0,
        m0,
    )

    # Right endpoint
    mN = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    mN = jnp.where(
        jnp.sign(mN) != jnp.sign(delta[-1]),
        0.0,
        mN,
    )
    mN = jnp.where(
        (jnp.sign(delta[-1]) != jnp.sign(delta[-2])) & (jnp.abs(mN) > jnp.abs(3.0 * delta[-1])),
        0.0,
        mN,
    )

    # assemble full slope array
    m = jnp.empty_like(y)
    m = m.at[0].set(m0)
    m = m.at[1:-1].set(m_inner)
    m = m.at[-1].set(mN)
    return m


def pchip_1d(x: jnp.ndarray,
             x_nodes: jnp.ndarray,
             y_nodes: jnp.ndarray) -> jnp.ndarray:
    """
    Monotone PCHIP interpolation y(x) for 1D data.

    x       : (...,)   eval points
    x_nodes : (N,)     knot positions (sorted)
    y_nodes : (N,)     values at knots

    Returns y(x) with same shape as x.
    """
    x = jnp.asarray(x)
    x_nodes = jnp.asarray(x_nodes)
    y_nodes = jnp.asarray(y_nodes)

    m_nodes = _pchip_slopes(x_nodes, y_nodes)  # (N,)

    # Find interval index i so that x in [x_i, x_{i+1}]
    idx = jnp.searchsorted(x_nodes, x, side="right") - 1
    nseg = x_nodes.shape[0] - 1
    idx = jnp.clip(idx, 0, nseg - 1)

    x0 = x_nodes[idx]
    x1 = x_nodes[idx + 1]
    y0 = y_nodes[idx]
    y1 = y_nodes[idx + 1]
    m0 = m_nodes[idx]
    m1 = m_nodes[idx + 1]

    h = x1 - x0
    t = (x - x0) / h

    # Cubic Hermite basis (same as before, but slopes are PCHIP slopes)
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2

    y = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
    return y
