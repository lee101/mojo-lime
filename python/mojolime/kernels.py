"""Public NumPy wrappers around the Mojo sampling kernels."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64, lib, parallel_runtime


def _matrix(value, name: str, *, nonempty: bool = True) -> np.ndarray:
    result = f64(value)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if nonempty and 0 in result.shape:
        raise ValueError(f"{name} must have at least one row and one column")
    return result


def _columns(value, columns: int, name: str) -> np.ndarray:
    result = f64(value)
    if result.shape != (columns,):
        raise ValueError(f"{name} must have one value per column")
    return result


def exponential_kernel(distances, kernel_width: float) -> np.ndarray:
    """LIME's default ``sqrt(exp(-d² / width²))`` proximity kernel."""
    values = f64(distances)
    width = float(kernel_width)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("kernel_width must be finite and greater than zero")
    result = np.empty(values.shape, dtype=np.float64, order="C")
    if values.size:
        lib().ml_kernel(addr(values), addr(result), values.size, width)
    return result


def affine_samples(values, center, scale) -> np.ndarray:
    """Apply per-column ``values * scale + center`` to a dense sample matrix."""
    values = _matrix(values, "values")
    center = _columns(center, values.shape[1], "center")
    scale = _columns(scale, values.shape[1], "scale")
    result = np.empty_like(values)
    lib().ml_affine(
        addr(values), addr(center), addr(scale), addr(result), *values.shape
    )
    return result


def affine_samples_inplace(values, center, scale) -> np.ndarray:
    values = _matrix(values, "values")
    center = _columns(center, values.shape[1], "center")
    scale = _columns(scale, values.shape[1], "scale")
    lib().ml_affine(
        addr(values), addr(center), addr(scale), addr(values), *values.shape
    )
    return values


def standardize(values, mean, scale) -> np.ndarray:
    """Apply per-column ``(values - mean) / scale``."""
    values = _matrix(values, "values")
    mean = _columns(mean, values.shape[1], "mean")
    scale = _columns(scale, values.shape[1], "scale")
    result = np.empty_like(values)
    lib().ml_standardize(
        addr(values), addr(mean), addr(scale), addr(result), *values.shape
    )
    return result


def standardize_inplace(values, mean, scale) -> np.ndarray:
    values = _matrix(values, "values")
    mean = _columns(mean, values.shape[1], "mean")
    scale = _columns(scale, values.shape[1], "scale")
    lib().ml_standardize(
        addr(values), addr(mean), addr(scale), addr(values), *values.shape
    )
    return values


def affine_standardize_samples(
    values, original, center, sample_scale, mean, standard_scale
):
    values = _matrix(values, "values")
    columns = values.shape[1]
    original = _columns(original, columns, "original")
    center = _columns(center, columns, "center")
    sample_scale = _columns(sample_scale, columns, "sample_scale")
    mean = _columns(mean, columns, "mean")
    standard_scale = _columns(standard_scale, columns, "standard_scale")
    inverse = np.empty_like(values)
    lib().ml_affine_standardize(
        addr(values),
        addr(original),
        addr(center),
        addr(sample_scale),
        addr(mean),
        addr(standard_scale),
        addr(inverse),
        *values.shape,
    )
    return values, inverse


def row_distances(values, metric: str = "euclidean", multiplier: float = 1.0) -> np.ndarray:
    """Distance from every dense row to row zero for LIME's common metrics."""
    values = _matrix(values, "values")
    result = np.empty(values.shape[0], dtype=np.float64)
    if metric == "euclidean":
        lib().ml_euclidean_rows(addr(values), addr(result), *values.shape)
        if multiplier != 1.0:
            result *= multiplier
    elif metric == "cosine":
        lib().ml_cosine_rows(
            addr(values), addr(result), *values.shape, float(multiplier)
        )
    else:
        raise ValueError("Mojo row_distances supports 'euclidean' and 'cosine'")
    return result


def image_neighborhood(image, fudged_image, segments, data) -> np.ndarray:
    """Materialize image perturbations selected by a binary sample matrix."""
    image = f64(image)
    fudged = f64(fudged_image)
    raw_segments = np.asarray(segments)
    if not np.issubdtype(raw_segments.dtype, np.integer):
        raise TypeError("segments must contain integers")
    if raw_segments.size and (
        np.any(raw_segments < 0)
        or np.any(raw_segments > np.iinfo(np.int64).max)
    ):
        raise ValueError("segments must be non-negative int64 values")
    segments = i64(raw_segments)
    data = _matrix(data, "data")
    if image.ndim != 3 or fudged.shape != image.shape:
        raise ValueError("image and fudged_image must have the same H x W x C shape")
    if segments.shape != image.shape[:2]:
        raise ValueError("segments must match the image's first two dimensions")
    if not np.all((data == 0) | (data == 1)):
        raise ValueError("data must be a binary matrix")
    if segments.size and np.max(segments) >= data.shape[1]:
        raise ValueError("segment IDs must index columns of data")
    result = np.empty((data.shape[0],) + image.shape, dtype=np.float64)
    lib().ml_image_neighborhood(
        addr(image),
        addr(fudged),
        addr(segments),
        addr(data),
        addr(result),
        data.shape[0],
        segments.size,
        image.shape[2],
        data.shape[1],
    )
    return result


def weighted_ridge(values, labels, weights, alpha: float = 1.0):
    """Fit a weighted ridge with intercept and return sklearn-shaped results.

    Returns ``(intercept, coef, score, local_prediction)`` where the local
    prediction is for row zero.
    """
    values = _matrix(values, "values")
    labels = f64(labels)
    weights = f64(weights)
    n, d = values.shape
    if labels.shape != (n,) or weights.shape != (n,):
        raise ValueError("labels and weights must have one entry per row")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(labels)):
        raise ValueError("values and labels must contain only finite values")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    coef = np.empty(d, dtype=np.float64)
    gram = np.empty((d, d), dtype=np.float64)
    rhs = np.empty(d, dtype=np.float64)
    means = np.empty(d, dtype=np.float64)
    work = n * d * (d + 1) // 2
    tasks = min(8, n) if work >= 2_000_000 else 1
    row = np.empty((tasks, d), dtype=np.float64)
    partial_gram = (
        np.empty((tasks, d, d), dtype=np.float64) if tasks > 1 else gram
    )
    partial_rhs = (
        np.empty((tasks, d), dtype=np.float64) if tasks > 1 else rhs
    )
    stats = np.empty(3, dtype=np.float64)
    library = lib()
    if tasks > 1:
        parallel_runtime()
    ok = library.ml_weighted_ridge(
        addr(values),
        addr(labels),
        addr(weights),
        addr(coef),
        addr(gram),
        addr(rhs),
        addr(means),
        addr(row),
        addr(partial_gram),
        addr(partial_rhs),
        addr(stats),
        n,
        d,
        alpha,
        tasks,
    )
    if not ok:
        raise np.linalg.LinAlgError("weighted ridge normal equations are not positive definite")
    return float(stats[0]), coef, float(stats[1]), np.array([stats[2]])
