"""Suite-wide JAX settings.

Full-precision float32 matmuls, everywhere in the tests. On Ampere and newer
GPUs JAX lowers float32 matmuls to TF32 by default, which carries about three
decimal digits of mantissa - and a good fraction of this suite asserts that two
orderings of the same computation agree (a manual forward pass against the
module, chunked features against unchunked, remat gradients against stored
ones). Those are transcription checks, not precision measurements; under TF32
they fail at 1e-3 on a GPU while passing on any CPU. Production code is not
affected: this file is loaded by pytest alone.
"""

import jax

jax.config.update("jax_default_matmul_precision", "highest")
