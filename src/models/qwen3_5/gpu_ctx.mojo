# ===----------------------------------------------------------------------=== #
# Module: src/models/qwen3_5/gpu_ctx.mojo
# Purpose: Penampung DeviceContext tunggal. Mojo melarang global variable,
#          maka context dialokasikan di heap dan pointer-nya dibagikan ke
#          seluruh layer. Satu context = buffer device bobot tetap valid dan
#          tidak ada release race (CUDA_ERROR_ILLEGAL_ADDRESS).
# ===----------------------------------------------------------------------=== #

from gpu.host import DeviceContext
from memory import UnsafePointer, alloc


fn gpu_ctx_new() raises -> UnsafePointer[DeviceContext, MutAnyOrigin]:
    """Alokasikan satu DeviceContext di heap (dipanggil sekali dari main)."""
    var p = alloc[DeviceContext](1)
    p.init_pointee_move(DeviceContext())
    return p


fn gpu_ctx_free(p: UnsafePointer[DeviceContext, MutAnyOrigin]):
    p.destroy_pointee()
    p.free()
