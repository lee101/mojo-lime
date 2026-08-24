# mojo-lime

LIME's local perturbation sampling and weighted surrogate fitting, implemented
in [Mojo](https://www.modular.com/mojo) and exposed through a Python API that
matches the covered upstream classes and method signatures.

This is a standalone implementation, not a wrapper around Python `lime`.
Upstream LIME is installed in the development environment only so the test
suite can compare both implementations on the same seeded samples and model
callbacks.

## Covered subset

| upstream module | implemented |
| --- | --- |
| `lime_base` | `LimeBase`, feature selection, and local weighted ridge fitting |
| `lime_tabular` | dense continuous and categorical neighborhoods, default quartile discretization, and classification/regression explanations |
| `lime_text` | `IndexedString`, `IndexedCharacters`, word bag-of-words sampling, and text explanations |
| `lime_image` | binary superpixel sampling, batched perturbed-image materialization, image explanations with custom segmentation, and `get_image_and_mask` |
| low-level kernels | proximity weights, affine sampling, standardization, Euclidean/cosine row distances, weighted ridge |

The covered constructors and methods retain upstream names, defaults, and
signatures under the `mojolime` package. The parity suite proves the behaviors
listed in the table against upstream LIME on seeded inputs.

Not covered: sparse tabular matrices, `RecurrentTabularExplainer`, HTML/notebook
rendering, LIME's submodular-pick module, non-local global explanation
workflows, and upstream behaviors not exercised by the parity suite. The source
contains additional compatible paths, including alternative discretizers,
distance metrics, character-level explanations, top-label selection, and
custom regressors, but they are not part of the tested support guarantee.
String assembly and classifier/model callbacks remain Python work; they are
not useful FFI kernel boundaries.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi install` brings the pinned Mojo nightly, Python dependencies, and
upstream `lime` used by parity tests. `pixi run build` creates
`dist/libmojo-lime.so`.

## Usage

```python
import numpy as np
from mojolime.lime_tabular import LimeTabularExplainer

rng = np.random.RandomState(0)
training = rng.normal(size=(1000, 4))
weights = np.array([1.5, -2.0, 0.5, 1.0])

def predict_proba(rows):
    probability = 1.0 / (1.0 + np.exp(-(rows @ weights)))
    return np.column_stack((1.0 - probability, probability))

explainer = LimeTabularExplainer(
    training,
    feature_names=["age", "balance", "visits", "tenure"],
    discretize_continuous=False,
    random_state=7,
)
result = explainer.explain_instance(
    training[0],
    predict_proba,
    labels=(1,),
    num_features=4,
    num_samples=5000,
)
print(result.as_list(label=1))
```

The example runs as written after `pixi install && pixi run build`.

## Performance

Measured on 2026-08-24 on an Intel Xeon E5-2697 v4 at 2.30GHz, Linux
6.8.0-136-generic. Times are the best of three warmed runs. The comparison uses
upstream LIME's actual NumPy/scikit-learn sampling paths, including its dense to
sparse conversion for text cosine distance. Run `pixi run bench` to reproduce
the table under a machine-wide benchmark lock.

| case | mojo-lime | upstream | result |
| --- | ---: | ---: | ---: |
| proximity kernel (5M distances) | 21.54 ms | 121.9 ms | 5.66x faster |
| Euclidean distances (100k x 64) | 5.19 ms | 22.97 ms | 4.43x faster |
| text cosine distances (100k x 128) | 21.95 ms | 703.7 ms | 32.06x faster |
| image `data_labels` (96 x 128 x 128 RGB) | 10.43 ms | 97.28 ms | 9.32x faster |
| weighted ridge surrogate (30k x 32) | 8.08 ms | 13.33 ms | 1.65x faster |
| tabular `explain_instance` (5k x 24) | 5.67 ms | 9.64 ms | 1.70x faster |

These are machine-specific results, not universal claims. The whole tabular
explanation gains less than the isolated distance kernels because prediction
callbacks and random-number generation are shared Python/NumPy work.

There is no GPU path. Distances and ridge updates do too little arithmetic per
byte transferred, while the higher-intensity Cholesky factorization operates on
only the small surrogate feature matrix. The elementwise proximity exponential
would require copying both its input and output for one operation, and the
pinned Mojo 1.1 SDK does not expose the former host-side `DeviceContext` launch
API. CPU remains the only supported device rather than shipping an unmeasured
GPU path.

## How it works

All Mojo numerical kernels are in one compilation unit. Python allocates
C-contiguous `float64` or `int64` NumPy arrays, then `ctypes` passes their
addresses and dimensions as C ABI integers. Mojo reconstructs pointers with a
mutable concrete origin, processes row-major buffers, and writes into
Python-owned output and scratch arrays. Nothing allocated in Python is retained
across an FFI call.

Sampling keeps NumPy's `RandomState` call order, which makes seeded
neighborhoods identical to upstream. Mojo applies the bulk affine transforms,
standardization, distance calculation, kernel weighting, image masks, and
weighted ridge normal equations. Continuous tabular sampling fuses its affine
and standardization passes. Large proximity transforms and Euclidean row sets
use eight CPU tasks, and only ridge fits large enough to amortize private
accumulators and task launch do the same; smaller calls stay serial. The first
sampled row is always the original instance, matching LIME's local-surrogate
contract.

## License

MIT
