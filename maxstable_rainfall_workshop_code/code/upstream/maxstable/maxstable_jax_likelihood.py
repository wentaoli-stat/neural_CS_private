"""JAX Smith max-stable pairwise log-likelihood.

This module implements the scalar Smith pairwise objective.  It uses the exact
standard-normal CDF required by the Smith density.  The default Jacobian is the
corrected GEV-to-unit-Frechet density correction implied by
SpatialExtremes::gev2frech; the pre-correction formula remains available for
audits with ``jacobian_mode="legacy"``.
"""

from __future__ import annotations

from functools import partial

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr


SMITH_SAFEGUARD_NAMES = (
    "xi_clip",
    "determinant_floor",
    "distance_floor",
    "scale_floor",
    "gev_support_floor",
    "frechet_floor",
    "density_floor",
    "replicate_nonfinite_replacement",
    "replicate_loglik_clip",
    "pair_total_loglik_clip",
)


def norm_cdf_jax(x):
    """Exact standard-normal CDF used by the Smith pairwise density."""

    return ndtr(x)


def norm_pdf_jax(x):
    """Standard normal PDF."""

    return 0.3989422804014327 * jnp.exp(-0.5 * x * x)


def max_stable_pairwise_loglik_multi_obs_jax(
    params,
    coord_i,
    coord_j,
    y_series_i,
    y_series_j,
    jacobian_mode="corrected",
):
    """Pairwise Smith max-stable log-likelihood for one site pair.

    Parameters
    ----------
    params:
        10-vector ``(cov11, cov12, cov22, beta_loc[3], beta_scale[3], xi)``.
    coord_i, coord_j:
        Two coordinate vectors of shape ``(2,)``.
    y_series_i, y_series_j:
        Replicate series for the two sites, each shape ``(n_obs,)``.

    jacobian_mode:
        ``"legacy"`` reproduces the pre-correction formula.
        ``"corrected"`` uses the GEV-to-unit-Frechet Jacobian implied by
        SpatialExtremes::gev2frech.

    Returns
    -------
    scalar JAX array
        Sum over replicates, clipped exactly like the Numba function.
    """

    params = jnp.asarray(params, dtype=jnp.float64)
    coord_i = jnp.asarray(coord_i, dtype=jnp.float64)
    coord_j = jnp.asarray(coord_j, dtype=jnp.float64)
    y_series_i = jnp.asarray(y_series_i, dtype=jnp.float64)
    y_series_j = jnp.asarray(y_series_j, dtype=jnp.float64)

    cov11, cov12, cov22 = params[0], params[1], params[2]
    beta_mu_0, beta_mu_1, beta_mu_2 = params[3], params[4], params[5]
    beta_la_0, beta_la_1, beta_la_2 = params[6], params[7], params[8]
    xi = jnp.clip(params[9], 0.01, 0.99)

    h_x = coord_j[0] - coord_i[0]
    h_y = coord_j[1] - coord_i[1]
    det_cov = jnp.maximum(cov11 * cov22 - cov12 * cov12, 1e-8)
    idet = 1.0 / det_cov
    a_squared = (cov22 * h_x * h_x - 2.0 * cov12 * h_x * h_y + cov11 * h_y * h_y) * idet
    a = jnp.sqrt(jnp.maximum(a_squared, 1e-8))
    a_safe = jnp.maximum(a, 1e-8)

    mu_i = beta_mu_0 + beta_mu_1 * coord_i[0] + beta_mu_2 * coord_i[1]
    mu_j = beta_mu_0 + beta_mu_1 * coord_j[0] + beta_mu_2 * coord_j[1]
    lambda_i = jnp.maximum(beta_la_0 + beta_la_1 * coord_i[0] + beta_la_2 * coord_i[1], 0.1)
    lambda_j = jnp.maximum(beta_la_0 + beta_la_1 * coord_j[0] + beta_la_2 * coord_j[1], 0.1)
    xi_safe = jnp.where(jnp.abs(xi) < 1e-6, 1e-6, xi)

    def one_replicate_loglik(y_i_t, y_j_t):
        gev_arg_i = jnp.maximum(1.0 + xi * (y_i_t - mu_i) / lambda_i, 1e-8)
        gev_arg_j = jnp.maximum(1.0 + xi * (y_j_t - mu_j) / lambda_j, 1e-8)

        z_i = jnp.maximum(jnp.power(gev_arg_i, 1.0 / xi_safe), 1e-8)
        z_j = jnp.maximum(jnp.power(gev_arg_j, 1.0 / xi_safe), 1e-8)

        w = a * 0.5 + jnp.log(z_j / z_i) / a
        v = a - w

        phi_w = norm_pdf_jax(w)
        phi_v = norm_pdf_jax(v)
        Phi_w = norm_cdf_jax(w)
        Phi_v = norm_cdf_jax(v)

        A = -Phi_w / z_i - Phi_v / z_j

        z_i2 = z_i * z_i
        z_j2 = z_j * z_j
        B = Phi_w / z_i2 + phi_w / (z_i2 * a_safe) - phi_v / (a_safe * z_j * z_i)
        C = Phi_v / z_j2 + phi_v / (z_j2 * a_safe) - phi_w / (a_safe * z_i * z_j)
        D = (
            v * phi_w / (a_safe * a_safe * z_i2 * z_j)
            + w * phi_v / (a_safe * a_safe * z_j2 * z_i)
        )
        BC_plus_D = jnp.maximum(B * C + D, 1e-12)

        if jacobian_mode == "legacy":
            E = (
                jnp.log(1.0 / (lambda_i * lambda_j))
                + jnp.log(gev_arg_i / lambda_i) * (1.0 / xi_safe - 1.0)
                + jnp.log(gev_arg_j / lambda_j) * (1.0 / xi_safe - 1.0)
            )
        elif jacobian_mode == "corrected":
            E = (
                -jnp.log(lambda_i)
                - jnp.log(lambda_j)
                + jnp.log(gev_arg_i) * (1.0 / xi_safe - 1.0)
                + jnp.log(gev_arg_j) * (1.0 / xi_safe - 1.0)
            )
        else:
            raise ValueError(f"Unknown jacobian_mode: {jacobian_mode}")

        loglik_t = jnp.clip(A + jnp.log(BC_plus_D) + E, -1000.0, 1000.0)
        return jnp.where(jnp.isfinite(loglik_t), loglik_t, 0.0)

    replicate_terms = jax.vmap(one_replicate_loglik)(y_series_i, y_series_j)
    total_loglik = jnp.sum(replicate_terms)
    return jnp.clip(total_loglik, -10000.0, 10000.0)


def max_stable_pairwise_safeguard_counts_jax(
    params,
    coord_i,
    coord_j,
    y_series_i,
    y_series_j,
    jacobian_mode="corrected",
):
    """Count every numerical safeguard used by the exact pair objective.

    This is a diagnostic mirror of the likelihood above; it does not alter the
    objective or its gradients.  It returns ``(counts, denominators)`` ordered
    by :data:`SMITH_SAFEGUARD_NAMES`, allowing formal preparation to report
    activation rates rather than silently relying on clipping/flooring.
    """

    params = jnp.asarray(params, dtype=jnp.float64)
    coord_i = jnp.asarray(coord_i, dtype=jnp.float64)
    coord_j = jnp.asarray(coord_j, dtype=jnp.float64)
    y_series_i = jnp.asarray(y_series_i, dtype=jnp.float64)
    y_series_j = jnp.asarray(y_series_j, dtype=jnp.float64)
    n_obs = y_series_i.shape[0]

    cov11, cov12, cov22 = params[0], params[1], params[2]
    beta_mu_0, beta_mu_1, beta_mu_2 = params[3], params[4], params[5]
    beta_la_0, beta_la_1, beta_la_2 = params[6], params[7], params[8]
    xi_raw = params[9]
    xi = jnp.clip(xi_raw, 0.01, 0.99)
    h_x = coord_j[0] - coord_i[0]
    h_y = coord_j[1] - coord_i[1]
    det_raw = cov11 * cov22 - cov12 * cov12
    det_cov = jnp.maximum(det_raw, 1e-8)
    a_squared_raw = (
        cov22 * h_x * h_x - 2.0 * cov12 * h_x * h_y + cov11 * h_y * h_y
    ) / det_cov
    a = jnp.sqrt(jnp.maximum(a_squared_raw, 1e-8))
    a_safe = jnp.maximum(a, 1e-8)
    mu_i = beta_mu_0 + beta_mu_1 * coord_i[0] + beta_mu_2 * coord_i[1]
    mu_j = beta_mu_0 + beta_mu_1 * coord_j[0] + beta_mu_2 * coord_j[1]
    lambda_i_raw = beta_la_0 + beta_la_1 * coord_i[0] + beta_la_2 * coord_i[1]
    lambda_j_raw = beta_la_0 + beta_la_1 * coord_j[0] + beta_la_2 * coord_j[1]
    lambda_i = jnp.maximum(lambda_i_raw, 0.1)
    lambda_j = jnp.maximum(lambda_j_raw, 0.1)
    xi_safe = jnp.where(jnp.abs(xi) < 1e-6, 1e-6, xi)

    def one_replicate(y_i_t, y_j_t):
        gev_i_raw = 1.0 + xi * (y_i_t - mu_i) / lambda_i
        gev_j_raw = 1.0 + xi * (y_j_t - mu_j) / lambda_j
        gev_i = jnp.maximum(gev_i_raw, 1e-8)
        gev_j = jnp.maximum(gev_j_raw, 1e-8)
        z_i_raw = jnp.power(gev_i, 1.0 / xi_safe)
        z_j_raw = jnp.power(gev_j, 1.0 / xi_safe)
        z_i = jnp.maximum(z_i_raw, 1e-8)
        z_j = jnp.maximum(z_j_raw, 1e-8)
        w = a * 0.5 + jnp.log(z_j / z_i) / a
        v = a - w
        phi_w = norm_pdf_jax(w)
        phi_v = norm_pdf_jax(v)
        Phi_w = norm_cdf_jax(w)
        Phi_v = norm_cdf_jax(v)
        A = -Phi_w / z_i - Phi_v / z_j
        z_i2 = z_i * z_i
        z_j2 = z_j * z_j
        B = Phi_w / z_i2 + phi_w / (z_i2 * a_safe) - phi_v / (
            a_safe * z_j * z_i
        )
        C = Phi_v / z_j2 + phi_v / (z_j2 * a_safe) - phi_w / (
            a_safe * z_i * z_j
        )
        D = (
            v * phi_w / (a_safe * a_safe * z_i2 * z_j)
            + w * phi_v / (a_safe * a_safe * z_j2 * z_i)
        )
        density_raw = B * C + D
        density = jnp.maximum(density_raw, 1e-12)
        if jacobian_mode == "legacy":
            E = (
                jnp.log(1.0 / (lambda_i * lambda_j))
                + jnp.log(gev_i / lambda_i) * (1.0 / xi_safe - 1.0)
                + jnp.log(gev_j / lambda_j) * (1.0 / xi_safe - 1.0)
            )
        elif jacobian_mode == "corrected":
            E = (
                -jnp.log(lambda_i)
                - jnp.log(lambda_j)
                + jnp.log(gev_i) * (1.0 / xi_safe - 1.0)
                + jnp.log(gev_j) * (1.0 / xi_safe - 1.0)
            )
        else:
            raise ValueError(f"Unknown jacobian_mode: {jacobian_mode}")
        raw_loglik = A + jnp.log(density) + E
        safe_loglik = jnp.where(
            jnp.isfinite(raw_loglik), jnp.clip(raw_loglik, -1000.0, 1000.0), 0.0
        )
        flags = jnp.asarray(
            (
                (gev_i_raw < 1e-8).astype(jnp.int64)
                + (gev_j_raw < 1e-8).astype(jnp.int64),
                (z_i_raw < 1e-8).astype(jnp.int64)
                + (z_j_raw < 1e-8).astype(jnp.int64),
                density_raw < 1e-12,
                ~jnp.isfinite(raw_loglik),
                jnp.abs(raw_loglik) > 1000.0,
            ),
            dtype=jnp.int64,
        )
        return safe_loglik, flags

    safe_terms, replicate_flags = jax.vmap(one_replicate)(
        y_series_i, y_series_j
    )
    replicate_counts = jnp.sum(replicate_flags, axis=0)
    total = jnp.sum(safe_terms)
    counts = jnp.asarray(
        (
            (xi_raw < 0.01) | (xi_raw > 0.99),
            det_raw < 1e-8,
            a_squared_raw < 1e-8,
            (lambda_i_raw < 0.1).astype(jnp.int64)
            + (lambda_j_raw < 0.1).astype(jnp.int64),
            replicate_counts[0],
            replicate_counts[1],
            replicate_counts[2],
            replicate_counts[3],
            replicate_counts[4],
            jnp.abs(total) > 10000.0,
        ),
        dtype=jnp.int64,
    )
    denominators = jnp.asarray(
        (1, 1, 1, 2, 2 * n_obs, 2 * n_obs, n_obs, n_obs, n_obs, 1),
        dtype=jnp.int64,
    )
    return counts, denominators


def make_all_pair_indices(n_locations: int):
    """Return all ``i < j`` pair indices as an ``(n_pairs, 2)`` int array."""

    pairs = []
    for i in range(int(n_locations)):
        for j in range(i + 1, int(n_locations)):
            pairs.append((i, j))
    return jnp.asarray(pairs, dtype=jnp.int32)


def max_stable_pairwise_loglik_pairs_jax(
    params,
    coords,
    y,
    pair_indices,
    jacobian_mode="corrected",
):
    """Composite pairwise log-likelihood summed over selected site pairs.

    ``pair_indices`` has shape ``(n_pairs, 2)`` and ``y`` has shape
    ``(n_obs, n_locations)``.
    """

    coords = jnp.asarray(coords, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    pair_indices = jnp.asarray(pair_indices, dtype=jnp.int32)

    def one_pair(pair):
        i, j = pair[0], pair[1]
        return max_stable_pairwise_loglik_multi_obs_jax(
            params,
            coords[i],
            coords[j],
            y[:, i],
            y[:, j],
            jacobian_mode=jacobian_mode,
        )

    return jnp.sum(jax.vmap(one_pair)(pair_indices))


@partial(jax.jit, static_argnames=("n_locations",))
def max_stable_pairwise_loglik_all_pairs_jax(
    params,
    coords,
    y,
    n_locations: int,
    jacobian_mode="corrected",
):
    """Composite pairwise log-likelihood summed over all site pairs.

    This is the JAX-native objective needed by NumPyro NUTS later. It assumes
    ``y`` has shape ``(n_obs, n_locations)``.
    """

    return max_stable_pairwise_loglik_pairs_jax(
        params,
        coords,
        y,
        make_all_pair_indices(n_locations),
        jacobian_mode=jacobian_mode,
    )
