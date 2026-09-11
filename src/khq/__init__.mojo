# ===----------------------------------------------------------------------=== #
# Module: src/khq/__init__.mojo — re-ekspor runtime KV terkompresi KHQ
# ===----------------------------------------------------------------------=== #
from .runtime import (
    khq_active, khq_activate, khq_init_layer, khq_capture_unroped, khq_step,
    khq_prof_report,
    KHQ_MAX_LAYERS, KHQ_RING, KHQ_WATERMARK, KHQ_CHUNK, KHQ_MAX_SPLITS,
)
