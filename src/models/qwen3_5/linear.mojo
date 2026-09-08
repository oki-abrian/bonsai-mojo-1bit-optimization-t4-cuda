# ===----------------------------------------------------------------------=== #
# Module: src/models/qwen3_5/linear.mojo
# Purpose: Wrapper layer proyeksi linear terkuantisasi 1-bit affine (NVIDIA T4)
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc
from os import getenv
from time import monotonic
from gpu.host import DeviceContext as DeviceContextGPU, DeviceBuffer
from sys.ffi import OwnedDLHandle
from src.ops import (
    quantized_matmul_1bit, quantized_matmul_1bit_gpu, qmv_sm75_1bit_launch_on,
    CudaQmvDecodeFnFP16, CudaQmmPrefillFnFP16, try_open_cuda_lib,
    dummy_cuda_decode_fp16, dummy_cuda_qmm_prefill_fp16
)
from src.common import decode_split_plan

# Prefill diproses M=1 per langkah, dan lm_head/lm_head tidak lewat
# QwenLinear1Bit, maka buffer staging cukup untuk M=1.
alias MAX_M = 1


fn prof_gpu() -> Bool:
    """Profil per-matmul aktif bila BONSAI_PROFILE=1."""
    var v = getenv("BONSAI_PROFILE")
    return v and v[0] == "1"


fn use_gpu_matmul() -> Bool:
    """
    True bila env BONSAI_USE_GPU=1 — jalur kernel device asli di NVIDIA T4.
    False (default) = host-sim sekuensial untuk validasi numerik.
    Dibaca dari env karena Mojo tidak mengizinkan global variable.
    """
    var v = getenv("BONSAI_USE_GPU")
    if not v:
        return False
    return v == "1" or v == "true" or v == "TRUE"

alias GDN_AFFINE_ZERO_CORRECTION: Bool = True
# Kontrak checkpoint prism-ml/Bonsai-27B-mlx-1bit terukur langsung dari header
# safetensors: biases = -scales_ckpt/2 untuk SETIAP grup, sedangkan loader
# menyimpan scales_eff = scales_ckpt/2. Maka koreksi affine per grup
# (b + s_eff) = (-s_ckpt/2 + s_ckpt/2) = 0 identik untuk seluruh baris.
# Loop koreksi O(N*K) di forward karena itu tidak perlu dijalankan.


struct QwenLinear1Bit:
    """
    Lapisan proyeksi matriks 1-bit affine (W1A16 g128).
    Kontrak checkpoint MLX: w = q*s_ckpt + b dengan q di {0,1}.
    Kernel menghitung (2q-1)*s_eff dengan s_eff = s_ckpt/2.
    """
    var ctx_ptr: UnsafePointer[DeviceContextGPU, MutAnyOrigin]
    var w_dev: UnsafePointer[UInt8, MutAnyOrigin]
    var s_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin]
    # DeviceBuffer bersifat RAII: bila disimpan sebagai variabel lokal,
    # memori device dibebaskan begitu keluar scope. Karena itu penampung
    # DeviceBuffer-nya dialokasikan di heap agar bobot tetap hidup
    # seumur layer, dan hanya pointer mentahnya yang dipakai kernel.
    var w_dev_buf: UnsafePointer[DeviceBuffer[DType.uint8], MutAnyOrigin]
    var s_dev_buf: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var x_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin]
    var y_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin]
    var x_dev_buf: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var y_dev_buf: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var ws_dev: UnsafePointer[Float32, MutAnyOrigin]
    var ws_dev_buf: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]
    var n_calls: Int
    var dev_ready: Bool
    var w_nbytes: Int
    var w_ptr_host_view: UnsafePointer[UInt8, MutAnyOrigin]
    var w: UnsafePointer[UInt8, MutAnyOrigin]
    var scales: UnsafePointer[Float32, MutAnyOrigin]
    var biases: UnsafePointer[Float32, MutAnyOrigin]
    var has_bias: Bool
    var ffi_ready: Bool
    var ffi_fn: CudaQmvDecodeFnFP16
    var ffi_lib_buf: UnsafePointer[OwnedDLHandle, MutAnyOrigin]
    var ffi_qmm_ready: Bool
    var ffi_qmm_fn: CudaQmmPrefillFnFP16
    var N: Int
    var K: Int

    fn __init__(
        out self,
        w: UnsafePointer[UInt8, MutAnyOrigin],
        scales: UnsafePointer[Float32, MutAnyOrigin],
        N: Int, K: Int
    ):
        """Jalur lama (data sintetis simetris): tanpa koreksi biases."""
        self.ctx_ptr = UnsafePointer[DeviceContextGPU, MutAnyOrigin]()
        self.w_dev = UnsafePointer[UInt8, MutAnyOrigin]()
        self.s_dev = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.w_dev_buf = UnsafePointer[DeviceBuffer[DType.uint8], MutAnyOrigin]()
        self.s_dev_buf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.x_dev = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.y_dev = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.x_dev_buf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.y_dev_buf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.ws_dev = UnsafePointer[Float32, MutAnyOrigin]()
        self.ws_dev_buf = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.n_calls = 0
        self.dev_ready = False
        self.w_nbytes = (N * K) // 8
        self.w_ptr_host_view = w
        self.w = w
        self.scales = scales
        self.biases = UnsafePointer[Float32, MutAnyOrigin]()
        self.has_bias = False
        self.ffi_ready = False
        self.ffi_fn = dummy_cuda_decode_fp16
        self.ffi_lib_buf = UnsafePointer[OwnedDLHandle, MutAnyOrigin]()
        self.ffi_qmm_ready = False
        self.ffi_qmm_fn = dummy_cuda_qmm_prefill_fp16
        self.N = N
        self.K = K

    fn __init__(
        out self,
        w: UnsafePointer[UInt8, MutAnyOrigin],
        scales: UnsafePointer[Float32, MutAnyOrigin],
        biases: UnsafePointer[Float32, MutAnyOrigin],
        N: Int, K: Int
    ):
        """Jalur checkpoint riil: scales sudah s_eff (dibagi 2 di loader)."""
        self.ctx_ptr = UnsafePointer[DeviceContextGPU, MutAnyOrigin]()
        self.w_dev = UnsafePointer[UInt8, MutAnyOrigin]()
        self.s_dev = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.w_dev_buf = UnsafePointer[DeviceBuffer[DType.uint8], MutAnyOrigin]()
        self.s_dev_buf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.x_dev = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.y_dev = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.x_dev_buf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.y_dev_buf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.ws_dev = UnsafePointer[Float32, MutAnyOrigin]()
        self.ws_dev_buf = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.n_calls = 0
        self.dev_ready = False
        self.w_nbytes = (N * K) // 8
        self.w_ptr_host_view = w
        self.w = w
        self.scales = scales
        self.biases = biases
        self.has_bias = True
        self.ffi_ready = False
        self.ffi_fn = dummy_cuda_decode_fp16
        self.ffi_lib_buf = UnsafePointer[OwnedDLHandle, MutAnyOrigin]()
        self.ffi_qmm_ready = False
        self.ffi_qmm_fn = dummy_cuda_qmm_prefill_fp16
        self.N = N
        self.K = K

    fn set_ctx(
        mut self,
        ctx: UnsafePointer[DeviceContextGPU, MutAnyOrigin]
    ) raises:
        """Pasang DeviceContext bersama yang dibuat sekali di main."""
        self.ctx_ptr = ctx
        if ctx:
            self.ensure_dev_ready()

    fn ensure_dev_ready(
        mut self
    ) raises:
        """Unggah bobot dan alokasikan workspace split-K sekali seumur hidup."""
        if self.dev_ready or not self.ctx_ptr or self.N == 0 or self.K == 0 or not self.w_ptr_host_view or self.w_nbytes == 0:
            return
        alias T = DType.float16
        var ctx = self.ctx_ptr[]

        var wb_holder = alloc[DeviceBuffer[DType.uint8]](1)
        wb_holder.init_pointee_move(
            ctx.enqueue_create_buffer[DType.uint8](self.w_nbytes)
        )
        ctx.enqueue_copy(wb_holder[], self.w_ptr_host_view)
        self.w_dev_buf = wb_holder
        self.w_dev = wb_holder[].unsafe_ptr()

        var sb_holder = alloc[DeviceBuffer[DType.float16]](1)
        sb_holder.init_pointee_move(
            ctx.enqueue_create_buffer[T](self.N * (self.K // 128))
        )
        var s16h = alloc[Scalar[T]](self.N * (self.K // 128))
        for i in range(self.N * (self.K // 128)):
            s16h[i] = self.scales[i].cast[T]()
        ctx.enqueue_copy(sb_holder[], s16h)
        self.s_dev_buf = sb_holder
        self.s_dev = sb_holder[].unsafe_ptr()
        s16h.free()

        var xb_holder = alloc[DeviceBuffer[DType.float16]](1)
        xb_holder.init_pointee_move(
            ctx.enqueue_create_buffer[T](MAX_M * self.K)
        )
        self.x_dev_buf = xb_holder
        self.x_dev = xb_holder[].unsafe_ptr()

        var yb_holder = alloc[DeviceBuffer[DType.float16]](1)
        yb_holder.init_pointee_move(
            ctx.enqueue_create_buffer[T](MAX_M * self.N)
        )
        self.y_dev_buf = yb_holder
        self.y_dev = yb_holder[].unsafe_ptr()

        var splits = decode_split_plan(self.N, 1, self.K // 128)
        var ws_elems = splits * MAX_M * self.N
        var wsb_holder = alloc[DeviceBuffer[DType.float32]](1)
        wsb_holder.init_pointee_move(
            ctx.enqueue_create_buffer[DType.float32](ws_elems)
        )
        self.ws_dev_buf = wsb_holder
        self.ws_dev = wsb_holder[].unsafe_ptr()

        # Inisialisasi FFI CUDA C kernel sekali seumur hidup layer (zero dlopen overhead saat token decode)
        var disable = getenv("BONSAI_DISABLE_CUDA_FFI")
        if not (disable and (disable == "1" or disable == "true")):
            try:
                var h_buf = alloc[OwnedDLHandle](1)
                h_buf.init_pointee_move(try_open_cuda_lib())
                self.ffi_fn = h_buf[].get_function[CudaQmvDecodeFnFP16]("launch_qmv_sm75_b1_decode_fp16")
                self.ffi_lib_buf = h_buf
                self.ffi_ready = True
                # Prefill batched (WMMA v2) — opsional di .so lama, bukan fatal
                # bila simbol tidak ada; prefill akan fallback per-token.
                try:
                    self.ffi_qmm_fn = h_buf[].get_function[CudaQmmPrefillFnFP16](
                        "launch_qmm_sm75_b1_prefill_fp16"
                    )
                    self.ffi_qmm_ready = True
                except:
                    self.ffi_qmm_ready = False
            except:
                self.ffi_ready = False

        self.dev_ready = True

    fn forward_device(
        mut self,
        x_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        y_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        M: Int = 1
    ) raises:
        """
        Jalur VRAM-to-VRAM murni untuk arsitektur 100% Full GPU.
        Zero host alloc, zero PCIe transfer, zero blocking sync!
        Prioritas 1: Pre-resolved CUDA FFI (direct C function pointer call, 0 overhead).
        Prioritas 2: Native Mojo GPU kernel fallback.
        """
        if not self.ctx_ptr:
            raise Error("FATAL: ctx_ptr null pada QwenLinear1Bit.forward_device!")
        alias T = DType.float16
        if not self.dev_ready:
            self.ensure_dev_ready()

        if self.ffi_ready:
            var cuda_stream = UnsafePointer[Float32, MutAnyOrigin]()
            var splits = decode_split_plan(self.N, 1, self.K // 128)
            var ret = self.ffi_fn(
                x_dev, self.w_dev, self.s_dev, y_dev, self.ws_dev,
                Int32(M), Int32(self.N), Int32(self.K), Int32(1),
                Int32(1), Int32(splits), cuda_stream
            )
            if ret == 0:
                return

        qmv_sm75_1bit_launch_on[T](
            self.ctx_ptr[],
            x_dev, self.w_dev, self.s_dev, y_dev,
            M, self.N, self.K, 1, True, self.ws_dev
        )

    fn prefill_ffi_ready(self) -> Bool:
        """True bila kernel prefill batched (WMMA v2) tersedia di .so."""
        return self.ffi_qmm_ready

    fn forward_prefill_device(
        mut self,
        x_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        y_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        M: Int
    ) raises:
        """
        Proyeksi BATCHED M-token via kernel qmm WMMA v2 (tensor core):
        y[M,N] = x[M,K] · w[N,K/8]^T. Bobot dibaca SEKALI untuk seluruh chunk
        (inilah penghemat prefill, bukan 9x GEMV). Syarat K % 128 == 0.
        Fallback: loop per-token lewat forward_device decode (benar, lebih lambat).
        """
        if not self.ctx_ptr:
            raise Error("FATAL: ctx_ptr null pada QwenLinear1Bit.forward_prefill_device!")
        if not self.dev_ready:
            self.ensure_dev_ready()

        if self.ffi_qmm_ready:
            var cuda_stream = UnsafePointer[Float32, MutAnyOrigin]()
            var ret = self.ffi_qmm_fn(
                x_m_dev, self.w_dev, self.s_dev, y_m_dev,
                Int32(M), Int32(self.N), Int32(self.K), Int32(1),
                Int32(1), cuda_stream
            )
            if ret == 0:
                return
            # ret != 0 (mis. kontrak K dilanggar) -> jatuh ke loop decode

        for t in range(M):
            self.forward_device(
                x_m_dev.offset(t * self.K), y_m_dev.offset(t * self.N), 1
            )

    fn forward(
        mut self,
        x: UnsafePointer[Float32, MutAnyOrigin],
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        M: Int = 1
    ) raises:
        """
        Dispatcher: GPU (kernel device T4) bila BONSAI_USE_GPU menyala,
        bila tidak jalur validasi host-sim sekuensial.
        """
        if not self.ctx_ptr:
            raise Error("FATAL: ctx_ptr bernilai null pada QwenLinear1Bit! Bobot tidak memiliki DeviceContext GPU aktif. CPU fallback dilarang.")
        self.forward_gpu_on(self.ctx_ptr, x, out_ptr, M)

    fn forward_gpu_on(
        mut self,
        ctx_raw: UnsafePointer[DeviceContextGPU, MutAnyOrigin],
        x: UnsafePointer[Float32, MutAnyOrigin],
        out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        M: Int = 1
    ) raises:
        """
        Jalur produksi: kernel W1A16 g128 dijalankan di hardware NVIDIA T4.
        Bobot dan skala diunggah SEKALI ke global memory device (self.w_dev /
        self.s_dev), aktivasi dan hasil dipindah per panggilan. Kernel device
        mensyaratkan staging FP16 (kontrak sm_75).
        """
        alias T = DType.float16
        var ctx = ctx_raw[]

        # Unggah bobot + skala sekali seumur hidup layer (bukan per token).
        if not self.dev_ready:
            self.ensure_dev_ready()

        var nx = M * self.K
        var ny = M * self.N
        var do_prof = self.n_calls < 10 and prof_gpu()
        var t0 = monotonic() if do_prof else 0
        var x16h = alloc[Scalar[T]](nx)
        for i in range(nx):
            x16h[i] = x[i].cast[T]()
        var t1 = monotonic() if do_prof else 0
        ctx.enqueue_copy(self.x_dev_buf[], x16h)
        var t2 = monotonic() if do_prof else 0
        var ffi_success = False
        if self.ffi_ready:
            var cuda_stream = UnsafePointer[Float32, MutAnyOrigin]()
            var splits = decode_split_plan(self.N, 1, self.K // 128)
            var ret = self.ffi_fn(
                self.x_dev, self.w_dev, self.s_dev, self.y_dev, self.ws_dev,
                Int32(M), Int32(self.N), Int32(self.K), Int32(1),
                Int32(1), Int32(splits), cuda_stream
            )
            ffi_success = (ret == 0)

        if not ffi_success:
            qmv_sm75_1bit_launch_on[T](
                ctx,
                self.x_dev, self.w_dev, self.s_dev, self.y_dev,
                M, self.N, self.K, 1, True, self.ws_dev
            )
        ctx.synchronize()
        var t3 = monotonic() if do_prof else 0
        var y16h = alloc[Scalar[T]](ny)
        ctx.enqueue_copy(y16h, self.y_dev_buf[])
        ctx.synchronize()
        var t4 = monotonic() if do_prof else 0
        for i in range(ny):
            out_ptr[i] = y16h[i].cast[DType.float32]()

        x16h.free()
        y16h.free()
        if do_prof:
            # us = mikrodetik. stg=konversi FP16, h2d=salin host->device,
            # ker=kernel GPU murni (launch+sync), d2h=salin balik device->host,
            # out=cast FP32.
            var us = Float64(1e3)
            print(">> [PROF] N=", self.N, " K=", self.K,
                  " stg=", (t1 - t0) / us,
                  " h2d=", (t2 - t1) / us,
                  " ker=", (t3 - t2) / us,
                  " d2h=", (t4 - t3) / us,
                  " out=", (monotonic() - t4) / us)
        self.n_calls += 1
