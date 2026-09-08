# ===----------------------------------------------------------------------=== #
# Module: src/models/qwen3_5/conv.mojo
# Purpose: Causal Depthwise 1D Convolution dengan Sliding Window State
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc
from .norm import silu

struct CausalConv1dState:
    """
    State buffer autoregresif untuk konvolusi 1D kausal (kernel_size = 4):
    Menyimpan 3 token riwayat sebelumnya [3, conv_dim] agar setiap token
    baru hanya memiliki akses informasi ke masa lalu (strictly causal).
    """
    var buffer: UnsafePointer[Float32, MutAnyOrigin] # [3 * conv_dim]
    var conv_dim: Int
    var kernel_size: Int

    fn __init__(out self, conv_dim: Int, kernel_size: Int = 4):
        self.conv_dim = conv_dim
        self.kernel_size = kernel_size
        self.buffer = alloc[Float32](3 * conv_dim)
        self.reset()

    fn reset(mut self):
        """Mengosongkan riwayat buffer konvolusi ke 0.0."""
        for i in range(3 * self.conv_dim):
            self.buffer[i] = 0.0

    fn step(
        self,
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        new_input: UnsafePointer[Float32, MutAnyOrigin],
        weights: UnsafePointer[Float32, MutAnyOrigin] # [conv_dim, 4]
    ):
        """
        Mengeksekusi 1 langkah causal depthwise 1D conv untuk token saat ini:
        conv_out[c] = silu( w[c,0]*buf[0,c] + w[c,1]*buf[1,c] + w[c,2]*buf[2,c] + w[c,3]*new_input[c] )
        lalu menggeser riwayat window: buf[0]=buf[1], buf[1]=buf[2], buf[2]=new_input.
        """
        for c in range(self.conv_dim):
            var b0 = self.buffer[c]
            var b1 = self.buffer[self.conv_dim + c]
            var b2 = self.buffer[2 * self.conv_dim + c]
            var b3 = new_input[c]

            var acc: Float32 = 0.0
            if weights:
                acc = b0 * weights[c * 4 + 0] + \
                      b1 * weights[c * 4 + 1] + \
                      b2 * weights[c * 4 + 2] + \
                      b3 * weights[c * 4 + 3]
            else:
                acc = (b0 + b1 + b2 + b3) * 0.25

            out_ptr[c] = silu(acc)

            # Geser riwayat kausal window
            self.buffer[c]                     = b1
            self.buffer[self.conv_dim + c]     = b2
            self.buffer[2 * self.conv_dim + c] = b3

    fn free(self):
        self.buffer.free()
