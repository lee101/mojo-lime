"""Numerical kernels for local surrogate explanation sampling.

Python owns every allocation. Buffers cross the C ABI as integer addresses and
are reconstructed here with a concrete mutable origin.
"""

from std.math import exp, sqrt
from std.sys import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def dot(a: Ptr, b: Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


def axpy(alpha: Float64, x: Ptr, y: Ptr, n: Int):
    var va = SIMD[DType.float64, W](alpha)
    var i = 0
    while i + W <= n:
        y.store(i, y.load[width=W](i) + va * x.load[width=W](i))
        i += W
    while i < n:
        y[i] += alpha * x[i]
        i += 1


def cholesky(a: Ptr, d: Int) -> Bool:
    for i in range(d):
        for j in range(i + 1):
            var value = a[i * d + j]
            for k in range(j):
                value -= a[i * d + k] * a[j * d + k]
            if i == j:
                if value <= 0.0:
                    return False
                a[i * d + i] = sqrt(value)
            else:
                a[i * d + j] = value / a[j * d + j]
    return True


def cholesky_solve(l: Ptr, b: Ptr, d: Int):
    for i in range(d):
        var value = b[i]
        for k in range(i):
            value -= l[i * d + k] * b[k]
        b[i] = value / l[i * d + i]
    for ri in range(d):
        var i = d - 1 - ri
        var value = b[i]
        for k in range(i + 1, d):
            value -= l[k * d + i] * b[k]
        b[i] = value / l[i * d + i]


def accumulate_ridge_parallel(
    x: Ptr,
    y: Ptr,
    w: Ptr,
    matrix: Ptr,
    target: Ptr,
    xmean: Ptr,
    row_work: Ptr,
    gram_work: Ptr,
    rhs_work: Ptr,
    n: Int,
    d: Int,
    ymean: Float64,
    tasks: Int,
):
    @parameter
    def accumulate(task: Int):
        var local_matrix = gram_work + task * d * d
        var local_target = rhs_work + task * d
        for j in range(d * d):
            local_matrix[j] = 0.0
        for j in range(d):
            local_target[j] = 0.0
        var start = task * n // tasks
        var end = (task + 1) * n // tasks
        for r in range(start, end):
            var wr = w[r]
            var ycentered = y[r] - ymean
            var j = 0
            while j + W <= d:
                var centered = (
                    x.load[width=W](r * d + j) -
                    xmean.load[width=W](j)
                )
                row_work.store(task * d + j, centered)
                local_target.store(
                    j,
                    local_target.load[width=W](j) +
                    SIMD[DType.float64, W](wr * ycentered) * centered,
                )
                j += W
            while j < d:
                row_work[task * d + j] = x[r * d + j] - xmean[j]
                local_target[j] += (
                    wr * row_work[task * d + j] * ycentered
                )
                j += 1
            var local_row = row_work + task * d
            for column in range(d):
                var factor = wr * local_row[column]
                if factor != 0.0:
                    axpy(
                        factor,
                        local_row,
                        local_matrix + column * d,
                        column + 1,
                    )

    for task in range(tasks):
        accumulate(task)
    for task in range(tasks):
        axpy(1.0, rhs_work + task * d, target, d)
        for column in range(d):
            axpy(
                1.0,
                gram_work + task * d * d + column * d,
                matrix + column * d,
                column + 1,
            )


@export("ml_kernel")
def ml_kernel(distances: Int, dst: Int, n: Int, width: Float64) abi("C"):
    var src = p(distances)
    var target = p(dst)
    var denom = 2.0 * width * width
    for i in range(n):
        var value = src[i]
        target[i] = exp(-(value * value) / denom)


@export("ml_affine")
def ml_affine(
    values: Int, center: Int, scale: Int, dst: Int, n: Int, d: Int
) abi("C"):
    var src = p(values)
    var centers = p(center)
    var scales = p(scale)
    var target = p(dst)
    for r in range(n):
        for j in range(d):
            target[r * d + j] = src[r * d + j] * scales[j] + centers[j]


@export("ml_standardize")
def ml_standardize(
    values: Int, mean: Int, scale: Int, dst: Int, n: Int, d: Int
) abi("C"):
    var src = p(values)
    var means = p(mean)
    var scales = p(scale)
    var target = p(dst)
    for r in range(n):
        for j in range(d):
            target[r * d + j] = (src[r * d + j] - means[j]) / scales[j]


@export("ml_affine_standardize")
def ml_affine_standardize(
    values: Int,
    original: Int,
    center: Int,
    sample_scale: Int,
    mean: Int,
    standard_scale: Int,
    inverse: Int,
    n: Int,
    d: Int,
) abi("C"):
    var data = p(values)
    var first = p(original)
    var centers = p(center)
    var sample_scales = p(sample_scale)
    var means = p(mean)
    var standard_scales = p(standard_scale)
    var unscaled_data = p(inverse)
    for r in range(n):
        var j = 0
        while j + W <= d:
            var unscaled: SIMD[DType.float64, W]
            if r == 0:
                unscaled = first.load[width=W](j)
            else:
                unscaled = (
                    data.load[width=W](r * d + j) *
                    sample_scales.load[width=W](j) +
                    centers.load[width=W](j)
                )
            unscaled_data.store(r * d + j, unscaled)
            data.store(
                r * d + j,
                (unscaled - means.load[width=W](j)) /
                standard_scales.load[width=W](j),
            )
            j += W
        while j < d:
            var unscaled = (
                first[j] if r == 0 else
                data[r * d + j] * sample_scales[j] + centers[j]
            )
            unscaled_data[r * d + j] = unscaled
            data[r * d + j] = (unscaled - means[j]) / standard_scales[j]
            j += 1


@export("ml_euclidean_rows")
def ml_euclidean_rows(values: Int, dst: Int, n: Int, d: Int) abi("C"):
    var x = p(values)
    var target = p(dst)
    for r in range(n):
        var row = x + r * d
        var acc = SIMD[DType.float64, W](0.0)
        var j = 0
        while j + W <= d:
            var diff = row.load[width=W](j) - x.load[width=W](j)
            acc += diff * diff
            j += W
        var total = acc.reduce_add()
        while j < d:
            var diff = row[j] - x[j]
            total += diff * diff
            j += 1
        target[r] = sqrt(total)


@export("ml_cosine_rows")
def ml_cosine_rows(
    values: Int, dst: Int, n: Int, d: Int, multiplier: Float64
) abi("C"):
    var x = p(values)
    var target = p(dst)
    var base_norm2 = dot(x, x, d)
    for r in range(n):
        var row = x + r * d
        var row_norm2 = dot(row, row, d)
        if base_norm2 == 0.0 or row_norm2 == 0.0:
            target[r] = 1.0 * multiplier
        else:
            var similarity = dot(row, x, d) / sqrt(row_norm2 * base_norm2)
            target[r] = (1.0 - similarity) * multiplier
    if n > 0:
        target[0] = 0.0


@export("ml_image_neighborhood")
def ml_image_neighborhood(
    image: Int,
    fudged: Int,
    segments: Int,
    data: Int,
    dst: Int,
    samples: Int,
    pixels: Int,
    channels: Int,
    features: Int,
) abi("C"):
    var original = p(image)
    var hidden = p(fudged)
    var segment_ids = ip(segments)
    var switches = p(data)
    var target = p(dst)
    for s in range(samples):
        var sample_offset = s * pixels * channels
        for pixel in range(pixels):
            var feature = Int(segment_ids[pixel])
            var use_original = (
                feature >= 0 and feature < features and
                switches[s * features + feature] != 0.0
            )
            var pixel_offset = pixel * channels
            for channel in range(channels):
                var idx = pixel_offset + channel
                target[sample_offset + idx] = (
                    original[idx] if use_original else hidden[idx]
                )


@export("ml_weighted_ridge")
def ml_weighted_ridge(
    values: Int,
    labels: Int,
    weights: Int,
    coef: Int,
    gram: Int,
    rhs: Int,
    means: Int,
    centered_row: Int,
    partial_gram: Int,
    partial_rhs: Int,
    stats: Int,
    n: Int,
    d: Int,
    alpha: Float64,
    tasks: Int,
) abi("C") -> Int:
    var x = p(values)
    var y = p(labels)
    var w = p(weights)
    var beta = p(coef)
    var matrix = p(gram)
    var target = p(rhs)
    var xmean = p(means)
    var row_work = p(centered_row)
    var gram_work = p(partial_gram)
    var rhs_work = p(partial_rhs)
    var result = p(stats)

    var sum_weight = 0.0
    var ymean = 0.0
    for j in range(d):
        xmean[j] = 0.0
    for r in range(n):
        var wr = w[r]
        sum_weight += wr
        ymean += wr * y[r]
        for j in range(d):
            xmean[j] += wr * x[r * d + j]
    if sum_weight <= 0.0:
        return 0
    ymean /= sum_weight
    for j in range(d):
        xmean[j] /= sum_weight
        target[j] = 0.0
    for j in range(d * d):
        matrix[j] = 0.0

    if tasks <= 1:
        for r in range(n):
            var wr = w[r]
            var ycentered = y[r] - ymean
            var j = 0
            while j + W <= d:
                var centered = x.load[width=W](r * d + j) - xmean.load[width=W](j)
                row_work.store(j, centered)
                target.store(
                    j,
                    target.load[width=W](j) +
                    SIMD[DType.float64, W](wr * ycentered) * centered,
                )
                j += W
            while j < d:
                row_work[j] = x[r * d + j] - xmean[j]
                target[j] += wr * row_work[j] * ycentered
                j += 1
            for column in range(d):
                var factor = wr * row_work[column]
                if factor != 0.0:
                    axpy(factor, row_work, matrix + column * d, column + 1)
    else:
        accumulate_ridge_parallel(
            x,
            y,
            w,
            matrix,
            target,
            xmean,
            row_work,
            gram_work,
            rhs_work,
            n,
            d,
            ymean,
            tasks,
        )

    for j in range(d):
        matrix[j * d + j] += alpha
        for k in range(j + 1, d):
            matrix[j * d + k] = matrix[k * d + j]
    if not cholesky(matrix, d):
        return 0
    cholesky_solve(matrix, target, d)
    for j in range(d):
        beta[j] = target[j]

    var intercept = ymean
    for j in range(d):
        intercept -= xmean[j] * beta[j]
    var local_prediction = intercept + dot(x, beta, d)
    var residual = 0.0
    var total = 0.0
    for r in range(n):
        var prediction = intercept + dot(x + r * d, beta, d)
        var error = y[r] - prediction
        var deviation = y[r] - ymean
        residual += w[r] * error * error
        total += w[r] * deviation * deviation
    var score: Float64
    if total == 0.0:
        score = 1.0 if residual == 0.0 else 0.0
    else:
        score = 1.0 - residual / total
    result[0] = intercept
    result[1] = score
    result[2] = local_prediction
    return 1
