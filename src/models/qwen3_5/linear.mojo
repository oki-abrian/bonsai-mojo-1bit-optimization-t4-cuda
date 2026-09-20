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
    CudaQmvDecodeFnFP16, CudaQmmPrefillFnFP16, CudaQmmPrefillB2FnFP16,
    CudaQmmPrefillB2FnINT8,
    CudaFwhtFnFP16,
    CudaQmvDenseFnFP16, try_open_cuda_lib,
    dummy_cuda_decode_fp16, dummy_cuda_qmm_prefill_fp16,
    dummy_cuda_qmm_prefill_b2_fp16, dummy_cuda_qmm_prefill_b2_int8,
    dummy_cuda_fwht_fp16,
    dummy_cuda_qmv_dense_fp16
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
    Lapisan proyeksi matriks kuantisasi affine W{n}A16 g128 (NVIDIA T4).

    bits=1 (Bonsai-27B lama): kontrak checkpoint MLX w = q*s_ckpt + b dengan
    q di {0,1}; kernel menghitung (2q-1)*s_eff dengan s_eff = s_ckpt/2.

    bits=2 (Bonsai-2 / Ternary-Bonsai-2-27B, Qwen3.8): kontrak pack MLX
    prism_hadamard_qwen35 — bobot U32 [N,K/16] (16 bobot/word, lane i di bit
    2i), biases == -scales penuh, sehingga w = (q-1)*s dan skala dipakai
    mentah. Modul terfold juga membawa vektor signs ±1 [K] dan memerlukan
    transformasi Hadamard blok-1024 pada AKTIVASI sebelum matmul (FWHT).
    Lihat CATATAN_IMPLEMENTASI_BONSAI2.md.
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
    # Prefill batched 2-bit ternary (WMMA v2). Terpisah dari ffi_qmm_fn karena
    # tanda tangannya berbeda: ada parameter n_total (stride baris output).
    var ffi_qmm2_ready: Bool
    var ffi_qmm2_fn: CudaQmmPrefillB2FnFP16
    # Prefill batched 2-bit ternary INT8 (tensor core int8 Turing, W2A8).
    # Jalur tambahan opt-in (BONSAI_PREFILL_INT8=1); tanda tangan identik
    # dengan ffi_qmm2_fn.
    var ffi_qmm2_int8_ready: Bool
    var ffi_qmm2_int8_fn: CudaQmmPrefillB2FnINT8
    # Decode 2-bit JALUR CEPAT (qmv_vec_q2t_h2_kernel: ekspansi kode lewat
    # PRMT + HFMA2 half2, bukan 6 instruksi skalar per bobot). Tanda tangan
    # identik dengan ffi_fn, jadi aliasnya sama. Opt-in lewat
    # BONSAI_DECODE_H2=1; default tetap jalur proven ffi_fn.
    var ffi_h2_ready: Bool
    var ffi_h2_fn: CudaQmvDecodeFnFP16
    var N: Int
    var K: Int
    # --- Bonsai-2 (bits=2) ---
    var bits: Int
    var signs: UnsafePointer[Float32, MutAnyOrigin]
    var signs_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin]
    var signs_dev_buf: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var has_signs: Bool
    var ffi_fwht_ready: Bool
    var ffi_fwht_fn: CudaFwhtFnFP16
    # --- Bonsai-2: ekor dense FP16 (modul tidak-terkuantisasi: in_proj_b/a) ---
    # N = N_packed + n_tail. Bagian packed (terkuantisasi 2-bit) menempati
    # baris output [0, N_packed); ekor dense F32 [n_tail, K] (tidak terfold,
    # tidak terkuantisasi di checkpoint) menempati [N_packed, N).
    var N_packed: Int
    var n_tail: Int
    var tail_w: UnsafePointer[Float32, MutAnyOrigin]
    var tail_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin]
    var tail_dev_buf: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var ffi_dense_ready: Bool
    var ffi_dense_fn: CudaQmvDenseFnFP16

    fn __init__(
        out self,
        w: UnsafePointer[UInt8, MutAnyOrigin],
        scales: UnsafePointer[Float32, MutAnyOrigin],
        N: Int, K: Int
    ):
        """Jalur lama (data sintetis simetris): tanpa koreksi biases. bits=1."""
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
        self.ffi_qmm2_ready = False
        self.ffi_qmm2_fn = dummy_cuda_qmm_prefill_b2_fp16
        self.ffi_qmm2_int8_ready = False
        self.ffi_qmm2_int8_fn = dummy_cuda_qmm_prefill_b2_int8
        self.ffi_h2_ready = False
        self.ffi_h2_fn = dummy_cuda_decode_fp16
        self.N = N
        self.K = K
        self.bits = 1
        self.signs = UnsafePointer[Float32, MutAnyOrigin]()
        self.signs_dev = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.signs_dev_buf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.has_signs = False
        self.ffi_fwht_ready = False
        self.ffi_fwht_fn = dummy_cuda_fwht_fp16
        self.N_packed = N
        self.n_tail = 0
        self.tail_w = UnsafePointer[Float32, MutAnyOrigin]()
        self.tail_dev = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.tail_dev_buf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.ffi_dense_ready = False
        self.ffi_dense_fn = dummy_cuda_qmv_dense_fp16

    fn __init__(
        out self,
        w: UnsafePointer[UInt8, MutAnyOrigin],
        scales: UnsafePointer[Float32, MutAnyOrigin],
        biases: UnsafePointer[Float32, MutAnyOrigin],
        N: Int, K: Int,
        bits: Int = 1,
        signs: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
        tail_w: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](),
        n_tail: Int = 0
    ):
        """
        Jalur checkpoint riil.
        bits=1: scales sudah s_eff (dibagi 2 di loader), kontrak (2q-1)*s.
        bits=2: scales mentah s_ckpt, kontrak (q-1)*s; signs ±1 [K] wajib untuk
                modul terfold (nil == modul tak terfold / vision tower).
        tail_w: ekor dense F32 [n_tail, K] untuk modul TIDAK terkuantisasi di
                pack Bonsai-2 (linear_attn.in_proj_b / in_proj_a). N harus
                SUDAH memasukkan n_tail (N = N_packed + n_tail). Ekornya
                dikalikan dengan input ASLI (tak ter-transform Hadamard).
        """
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
        self.N = N
        self.n_tail = n_tail
        self.N_packed = N - n_tail
        # 1-bit: 8 bobot/byte; 2-bit: 4 bobot/byte. Hanya bagian packed.
        self.w_nbytes = (self.N_packed * K) // (8 // bits)
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
        self.ffi_qmm2_ready = False
        self.ffi_qmm2_fn = dummy_cuda_qmm_prefill_b2_fp16
        self.ffi_qmm2_int8_ready = False
        self.ffi_qmm2_int8_fn = dummy_cuda_qmm_prefill_b2_int8
        self.ffi_h2_ready = False
        self.ffi_h2_fn = dummy_cuda_decode_fp16
        self.K = K
        self.bits = bits
        self.signs = signs
        self.signs_dev = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.signs_dev_buf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.has_signs = signs != UnsafePointer[Float32, MutAnyOrigin]()
        self.ffi_fwht_ready = False
        self.ffi_fwht_fn = dummy_cuda_fwht_fp16
        self.tail_w = tail_w
        self.tail_dev = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.tail_dev_buf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.ffi_dense_ready = False
        self.ffi_dense_fn = dummy_cuda_qmv_dense_fp16

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
            ctx.enqueue_create_buffer[T](self.N_packed * (self.K // 128))
        )
        var s16h = alloc[Scalar[T]](self.N_packed * (self.K // 128))
        for i in range(self.N_packed * (self.K // 128)):
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

        var splits = decode_split_plan(self.N_packed, 1, self.K // 128)
        var ws_elems = splits * MAX_M * self.N_packed
        var wsb_holder = alloc[DeviceBuffer[DType.float32]](1)
        wsb_holder.init_pointee_move(
            ctx.enqueue_create_buffer[DType.float32](ws_elems)
        )
        self.ws_dev_buf = wsb_holder
        self.ws_dev = wsb_holder[].unsafe_ptr()

        # Bonsai-2: unggah vektor signs ±1 [K] sebagai F16 (±1 eksak di F16).
        if self.has_signs and self.signs:
            var sg_holder = alloc[DeviceBuffer[DType.float16]](1)
            sg_holder.init_pointee_move(
                ctx.enqueue_create_buffer[T](self.K)
            )
            var s16g = alloc[Scalar[T]](self.K)
            for i in range(self.K):
                s16g[i] = self.signs[i].cast[T]()
            ctx.enqueue_copy(sg_holder[], s16g)
            self.signs_dev_buf = sg_holder
            self.signs_dev = sg_holder[].unsafe_ptr()
            s16g.free()

        # Bonsai-2: unggah ekor dense F32 [n_tail, K] sebagai F16.
        if self.n_tail > 0 and self.tail_w:
            var tw_holder = alloc[DeviceBuffer[DType.float16]](1)
            tw_holder.init_pointee_move(
                ctx.enqueue_create_buffer[T](self.n_tail * self.K)
            )
            var t16w = alloc[Scalar[T]](self.n_tail * self.K)
            for i in range(self.n_tail * self.K):
                t16w[i] = self.tail_w[i].cast[T]()
            ctx.enqueue_copy(tw_holder[], t16w)
            self.tail_dev_buf = tw_holder
            self.tail_dev = tw_holder[].unsafe_ptr()
            t16w.free()

        # Inisialisasi FFI CUDA C kernel sekali seumur hidup layer (zero dlopen overhead saat token decode)
        var disable = getenv("BONSAI_DISABLE_CUDA_FFI")
        if not (disable and (disable == "1" or disable == "true")):
            try:
                var h_buf = alloc[OwnedDLHandle](1)
                h_buf.init_pointee_move(try_open_cuda_lib())
                # Pilih simbol GEMV sesuai bit-width: b1 untuk 1-bit, b2 untuk
                # 2-bit ternary (tanda tangan FFI identik -> alias sama).
                var sym = "launch_qmv_sm75_b1_decode_fp16" if self.bits == 1 else "launch_qmv_sm75_b2_decode_fp16"
                self.ffi_fn = h_buf[].get_function[CudaQmvDecodeFnFP16](sym)
                self.ffi_lib_buf = h_buf
                self.ffi_ready = True
                # Prefill batched (WMMA v2): simbol 1-bit untuk bits=1,
                # simbol 2-bit untuk bits=2. Bukan fatal bila simbol tak ada
                # di .so lama — prefill lalu mundur ke loop per-token.
                if self.bits == 1:
                    try:
                        self.ffi_qmm_fn = h_buf[].get_function[CudaQmmPrefillFnFP16](
                            "launch_qmm_sm75_b1_prefill_fp16"
                        )
                        self.ffi_qmm_ready = True
                    except:
                        self.ffi_qmm_ready = False
                else:
                    self.ffi_qmm_ready = False
                    try:
                        self.ffi_qmm2_fn = h_buf[].get_function[CudaQmmPrefillB2FnFP16](
                            "launch_qmm_sm75_b2_prefill_fp16"
                        )
                        self.ffi_qmm2_ready = True
                    except:
                        self.ffi_qmm2_ready = False
                    # Jalur int8 tensor core (W2A8). Hanya di-resolve bila
                    # BONSAI_PREFILL_INT8=1: jalur ini mengubah basis numerik
                    # (aktivasi dikuantisasi ke int8), sehingga default tetap
                    # jalur WMMA fp16 yang sudah terverifikasi. Simbol tak ada
                    # di .so lama -> tidak fatal, ready=False.
                    var want_int8 = getenv("BONSAI_PREFILL_INT8")
                    if want_int8 and (want_int8 == "1" or want_int8 == "true"):
                        try:
                            self.ffi_qmm2_int8_fn = h_buf[].get_function[
                                CudaQmmPrefillB2FnINT8
                            ]("launch_qmm_sm75_b2_prefill_int8")
                            self.ffi_qmm2_int8_ready = True
                            # Tanda terlihat: tanpa baris ini suatu run bisa
                            # tanpa suara jatuh ke jalur fp16 (simbol tak ada
                            # atau env tak terbaca) dan terlihat "lulus uji"
                            # tanpa benar-benar menguji int8.
                            print(
                                ">> [INT8] jalur prefill int8 (W2A8) aktif, " +
                                "N=" + String(self.N) + " K=" + String(self.K)
                            )
                        except:
                            print(
                                ">> [INT8] PERHATIAN: BONSAI_PREFILL_INT8=1 " +
                                "tapi simbol launch_qmm_sm75_b2_prefill_int8 " +
                                "tidak ada di .so — jatuh ke jalur WMMA fp16"
                            )
                            self.ffi_qmm2_int8_ready = False
                # Decode 2-bit JALUR CEPAT: qmv_vec_q2t_h2_kernel (ekspansi
                # kode lewat PRMT + HFMA2 half2). Opt-in (BONSAI_DECODE_H2=1)
                # karena ia mengubah URUTAN PENJUMLAHAN internal (akumulasi
                # half2 tiap 8 bobot, lalu FP32), jadi tidak boleh menggantikan
                # jalur proven ffi_fn secara diam-diam.
                var want_h2 = getenv("BONSAI_DECODE_H2")
                if self.bits == 2 and want_h2 and (
                    want_h2 == "1" or want_h2 == "true" or want_h2 == "2"
                ):
                    # "2" = varian LEBAR (qmv_vec_q2t_h2b_kernel: 2 grup per
                    # lane, 4x uint4 diterbitkan sebelum dihitung). Tanda
                    # tangan identik, jadi cukup memilih simbolnya — ini
                    # memungkinkan A/B varian 1 vs 2 dalam SATU build.
                    # Nama simbol ditulis sebagai LITERAL di tiap cabang,
                    # bukan disimpan di variabel: get_function dipanggil
                    # dengan literal di seluruh codebase ini, dan build
                    # jarak jauh tidak boleh gagal karena tipe argumen.
                    if want_h2 == "2":
                        try:
                            self.ffi_h2_fn = h_buf[].get_function[
                                CudaQmvDecodeFnFP16
                            ]("launch_qmv_sm75_b2_decode_h2b")
                            self.ffi_h2_ready = True
                            print(
                                ">> [H2] jalur decode cepat VARIAN LEBAR " +
                                "(h2b, 2 grup/lane) aktif, N=" +
                                String(self.N_packed) + " K=" + String(self.K)
                            )
                        except:
                            self.ffi_h2_ready = False
                            print(
                                ">> [H2] PERHATIAN: BONSAI_DECODE_H2=2 tapi " +
                                "simbol launch_qmv_sm75_b2_decode_h2b tidak " +
                                "ada di .so — jatuh ke jalur decode fp16 lama"
                            )
                    else:
                        try:
                            self.ffi_h2_fn = h_buf[].get_function[
                                CudaQmvDecodeFnFP16
                            ]("launch_qmv_sm75_b2_decode_h2")
                            self.ffi_h2_ready = True
                            # Tanda terlihat: seperti jalur int8, tanpa baris
                            # ini suatu run bisa "lulus uji" tanpa benar-benar
                            # menguji jalur baru (simbol tak ada / env tak
                            # terbaca).
                            print(
                                ">> [H2] jalur decode cepat (PRMT+HFMA2) " +
                                "aktif, N=" + String(self.N_packed) +
                                " K=" + String(self.K)
                            )
                        except:
                            self.ffi_h2_ready = False
                            print(
                                ">> [H2] PERHATIAN: BONSAI_DECODE_H2=1 tapi " +
                                "simbol launch_qmv_sm75_b2_decode_h2 tidak " +
                                "ada di .so — jatuh ke jalur decode fp16 lama"
                            )
                # FWHT Hadamard activation transform (Bonsai-2, modul terfold).
                if self.has_signs:
                    try:
                        self.ffi_fwht_fn = h_buf[].get_function[CudaFwhtFnFP16](
                            "launch_fwht_sm75_fp16"
                        )
                        self.ffi_fwht_ready = True
                    except:
                        self.ffi_fwht_ready = False
                # GEMV dense untuk ekor FP32 tidak-terkuantisasi (Bonsai-2).
                if self.n_tail > 0:
                    try:
                        self.ffi_dense_fn = h_buf[].get_function[CudaQmvDenseFnFP16](
                            "launch_qmv_sm75_dense_fp16"
                        )
                        self.ffi_dense_ready = True
                    except:
                        self.ffi_dense_ready = False
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

        # 2-bit ternary hanya didukung lewat FFI CUDA (kernel Mojo native
        # hanya ada untuk 1-bit). Menolak keras lebih baik daripada output
        # salah diam-diam.
        if self.bits != 1 and not self.ffi_ready:
            raise Error(
                "bits=2 memerlukan libbonsai_qmv_sm75.so dengan simbol " +
                "launch_qmv_sm75_b2_decode_fp16 (FFI tidak aktif)"
            )
        # Ekor dense (in_proj_b/a) wajib ada kernelnya untuk bits=2.
        if self.n_tail > 0 and not self.ffi_dense_ready:
            raise Error(
                "modul Bonsai-2 dengan ekor dense (in_proj_b/a) memerlukan " +
                "simbol launch_qmv_sm75_dense_fp16 di libbonsai_qmv_sm75.so"
            )

        # Bonsai-2: Hadamard activation transform pada input. x_dev sering
        # di-SHARE antar beberapa proyeksi (mis. q/k/v memakai x_norm_dev
        # yang sama), jadi hasil transformasi ditulis ke buffer privat
        # self.x_dev — sekaligus berfungsi sebagai salin, mencegah
        # double-transform bila proyeksi berikutnya memakai x_dev mentah.
        # Tanpa signs (1-bit / modul tak terfold): pakai x_dev langsung.
        var x_eff = x_dev
        if self.has_signs:
            if not self.ffi_fwht_ready:
                raise Error(
                    "modul terfold Bonsai-2 memerlukan kernel FWHT " +
                    "(launch_fwht_sm75_fp16) di libbonsai_qmv_sm75.so"
                )
            x_eff = self.x_dev
            var fwht_stream = UnsafePointer[Float32, MutAnyOrigin]()
            var fret = self.ffi_fwht_fn(
                x_dev, self.x_dev, self.signs_dev,
                Int32(M), Int32(self.K), Int32(0), fwht_stream
            )
            if fret != 0:
                raise Error(
                    "FWHT forward gagal: ret=" + String(Int(fret))
                    + " (N=" + String(self.N) + " K=" + String(self.K) + ")"
                )

        # Bagian packed (terkuantisasi): N_packed baris pertama output.
        if self.ffi_ready:
            var cuda_stream = UnsafePointer[Float32, MutAnyOrigin]()
            var splits = decode_split_plan(self.N_packed, 1, self.K // 128)
            # Jalur cepat half2 bila diaktifkan (BONSAI_DECODE_H2=1) dan
            # simbolnya ada; kalau tidak, persis kode lama. Tanda tangan
            # keduanya identik.
            var ret: Int32
            if self.ffi_h2_ready:
                ret = self.ffi_h2_fn(
                    x_eff, self.w_dev, self.s_dev, y_dev, self.ws_dev,
                    Int32(M), Int32(self.N_packed), Int32(self.K), Int32(1),
                    Int32(1), Int32(splits), cuda_stream
                )
            else:
                ret = self.ffi_fn(
                    x_eff, self.w_dev, self.s_dev, y_dev, self.ws_dev,
                    Int32(M), Int32(self.N_packed), Int32(self.K), Int32(1),
                    Int32(1), Int32(splits), cuda_stream
                )
            if ret == 0:
                # Ekor dense: input ASLI x_dev (modul tidak terfold ->
                # TIDAK boleh memakai x_eff yang sudah di-FWHT).
                if self.n_tail > 0:
                    var dstream = UnsafePointer[Float32, MutAnyOrigin]()
                    var dret = self.ffi_dense_fn(
                        x_dev, self.tail_dev, y_dev,
                        Int32(self.N_packed), Int32(self.N), Int32(self.n_tail),
                        Int32(self.K), Int32(M), dstream
                    )
                    if dret != 0:
                        raise Error(
                            "kernel GEMV dense gagal: ret=" + String(Int(dret))
                            + " (n_tail=" + String(self.n_tail)
                            + " K=" + String(self.K) + ")"
                        )
                return
            if self.bits != 1:
                raise Error(
                    "kernel GEMV 2-bit gagal: ret=" + String(Int(ret))
                    + " (N=" + String(self.N) + " K=" + String(self.K) + ")"
                )
            # ret != 0 (kontrak dilanggar) -> jatuh ke fallback Mojo native 1-bit

        qmv_sm75_1bit_launch_on[T](
            self.ctx_ptr[],
            x_eff, self.w_dev, self.s_dev, y_dev,
            M, self.N_packed, self.K, 1, True, self.ws_dev
        )
        # Ekor dense pada jalur fallback Mojo native. Hanya tercapai untuk
        # bits=1 (bits=2 raise di atas); n_tail > 0 hanya untuk bits=2, jadi
        # loop ini no-op pada pemakaian nyata — tetap pakai FFI agar tidak
        # membaca pointer device dari host.
        if self.n_tail > 0 and self.ffi_dense_ready:
            var dstream2 = UnsafePointer[Float32, MutAnyOrigin]()
            self.ffi_dense_fn(
                x_dev, self.tail_dev, y_dev,
                Int32(self.N_packed), Int32(self.N), Int32(self.n_tail),
                Int32(self.K), Int32(M), dstream2
            )

    fn prefill_ffi_ready(self) -> Bool:
        """True bila kernel prefill batched tersedia di .so (WMMA v2 atau int8)."""
        return self.ffi_qmm_ready or self.ffi_qmm2_ready or self.ffi_qmm2_int8_ready

    fn forward_prefill_per_token(
        mut self,
        x_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        y_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        M: Int
    ) raises:
        """Fallback prefill: loop decode per-token (benar, lebih lambat)."""
        for t in range(M):
            self.forward_device(
                x_m_dev.offset(t * self.K), y_m_dev.offset(t * self.N), 1
            )

    fn forward_prefill_device(
        mut self,
        x_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        y_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        M: Int,
        fwht_scratch_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin] = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
    ) raises:
        """
        Proyeksi BATCHED M-token via kernel qmm WMMA v2 (tensor core):
        y[M,N] = x[M,K] · w[N,K/8]^T. Bobot dibaca SEKALI untuk seluruh chunk
        (inilah penghemat prefill, bukan 9x GEMV). Syarat K % 128 == 0.

        bits=2: aktivasi harus di-FWHT lebih dulu (modul terfold) dan hasilnya
        ditulis ke buffer scratch milik pemanggil — bukan ke x_m_dev, sebab
        buffer itu dipakai bersama beberapa proyeksi (q/k/v). Bila scratch
        tidak disediakan, mundur ke loop per-token.
        Fallback: loop per-token lewat forward_device decode (benar, lebih lambat).
        """
        if not self.ctx_ptr:
            raise Error("FATAL: ctx_ptr null pada QwenLinear1Bit.forward_prefill_device!")
        if not self.dev_ready:
            self.ensure_dev_ready()

        # ---- Jalur 2-bit ternary int8 (tensor core int8 Turing, W2A8) ----
        # Opt-in via BONSAI_PREFILL_INT8=1. Basis numerik berbeda dari WMMA
        # fp16 (aktivasi dikuantisasi ke int8), jadi uji A/B prefill vs
        # per-token wajib diulang bila jalur ini diaktifkan. Ret != 0 atau
        # simbol tak ada -> jatuh ke jalur WMMA fp16 di bawah, lalu per-token.
        if self.bits == 2 and self.ffi_qmm2_int8_ready:
            var x_eff2 = x_m_dev
            if self.has_signs:
                if not self.ffi_fwht_ready:
                    raise Error(
                        "modul terfold Bonsai-2 memerlukan kernel FWHT "
                        + "(launch_fwht_sm75_fp16) di libbonsai_qmv_sm75.so"
                    )
                if fwht_scratch_dev == UnsafePointer[Scalar[DType.float16], MutAnyOrigin]():
                    self.forward_prefill_per_token(x_m_dev, y_m_dev, M)
                    return
                var fstream2 = UnsafePointer[Float32, MutAnyOrigin]()
                var fret2 = self.ffi_fwht_fn(
                    x_m_dev, fwht_scratch_dev, self.signs_dev,
                    Int32(M), Int32(self.K), Int32(0), fstream2
                )
                if fret2 != 0:
                    raise Error(
                        "FWHT forward prefill gagal: ret=" + String(Int(fret2))
                        + " (N=" + String(self.N) + " K=" + String(self.K) + ")"
                    )
                x_eff2 = fwht_scratch_dev

            var qstream2 = UnsafePointer[Float32, MutAnyOrigin]()
            var ret2 = self.ffi_qmm2_int8_fn(
                x_eff2, self.w_dev, self.s_dev, y_m_dev,
                Int32(M), Int32(self.N_packed), Int32(self.N), Int32(self.K),
                Int32(1), Int32(1), qstream2
            )
            if ret2 == 0:
                # Ekor dense: input ASLI (modul tidak terfold -> TIDAK boleh
                # memakai hasil FWHT). Kernel dense sudah mendukung L baris.
                if self.n_tail > 0:
                    if not self.ffi_dense_ready:
                        raise Error(
                            "ekor dense Bonsai-2 memerlukan simbol "
                            + "launch_qmv_sm75_dense_fp16 di libbonsai_qmv_sm75.so"
                        )
                    var dstream2 = UnsafePointer[Float32, MutAnyOrigin]()
                    var dret2 = self.ffi_dense_fn(
                        x_m_dev, self.tail_dev, y_m_dev,
                        Int32(self.N_packed), Int32(self.N), Int32(self.n_tail),
                        Int32(self.K), Int32(M), dstream2
                    )
                    if dret2 != 0:
                        raise Error(
                            "kernel GEMV dense prefill gagal: ret="
                            + String(Int(dret2))
                            + " (n_tail=" + String(self.n_tail)
                            + " K=" + String(self.K) + ")"
                        )
                return
            # ret2 != 0 (mis. kontrak K dilanggar) -> jatuh ke jalur WMMA fp16

        # ---- Jalur 2-bit ternary (WMMA v2, bobot dibaca sekali) ----
        if self.bits == 2 and self.ffi_qmm2_ready:
            var x_eff2 = x_m_dev
            if self.has_signs:
                if not self.ffi_fwht_ready:
                    raise Error(
                        "modul terfold Bonsai-2 memerlukan kernel FWHT " +
                        "(launch_fwht_sm75_fp16) di libbonsai_qmv_sm75.so"
                    )
                if fwht_scratch_dev == UnsafePointer[Scalar[DType.float16], MutAnyOrigin]():
                    self.forward_prefill_per_token(x_m_dev, y_m_dev, M)
                    return
                var fstream2 = UnsafePointer[Float32, MutAnyOrigin]()
                var fret2 = self.ffi_fwht_fn(
                    x_m_dev, fwht_scratch_dev, self.signs_dev,
                    Int32(M), Int32(self.K), Int32(0), fstream2
                )
                if fret2 != 0:
                    raise Error(
                        "FWHT forward prefill gagal: ret=" + String(Int(fret2))
                        + " (N=" + String(self.N) + " K=" + String(self.K) + ")"
                    )
                x_eff2 = fwht_scratch_dev

            var qstream2 = UnsafePointer[Float32, MutAnyOrigin]()
            var ret2 = self.ffi_qmm2_fn(
                x_eff2, self.w_dev, self.s_dev, y_m_dev,
                Int32(M), Int32(self.N_packed), Int32(self.N), Int32(self.K),
                Int32(1), Int32(1), qstream2
            )
            if ret2 == 0:
                # Ekor dense: input ASLI (modul tidak terfold -> TIDAK boleh
                # memakai hasil FWHT). Kernel dense sudah mendukung L baris.
                if self.n_tail > 0:
                    if not self.ffi_dense_ready:
                        raise Error(
                            "ekor dense Bonsai-2 memerlukan simbol " +
                            "launch_qmv_sm75_dense_fp16 di libbonsai_qmv_sm75.so"
                        )
                    var dstream2 = UnsafePointer[Float32, MutAnyOrigin]()
                    var dret2 = self.ffi_dense_fn(
                        x_m_dev, self.tail_dev, y_m_dev,
                        Int32(self.N_packed), Int32(self.N), Int32(self.n_tail),
                        Int32(self.K), Int32(M), dstream2
                    )
                    if dret2 != 0:
                        raise Error(
                            "kernel GEMV dense prefill gagal: ret="
                            + String(Int(dret2))
                            + " (n_tail=" + String(self.n_tail)
                            + " K=" + String(self.K) + ")"
                        )
                return
            # ret2 != 0 (mis. kontrak K dilanggar) -> jatuh ke loop per-token

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

        self.forward_prefill_per_token(x_m_dev, y_m_dev, M)

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

        # Bonsai-2: Hadamard activation transform in-place di buffer privat
        # self.x_dev (buffer ini milik layer ini saja -> aman dari
        # double-transform; kernel FWHT membaca dan menulis global 1x).
        if self.has_signs:
            if not self.ffi_fwht_ready:
                x16h.free()
                raise Error(
                    "modul terfold Bonsai-2 memerlukan kernel FWHT " +
                    "(launch_fwht_sm75_fp16) di libbonsai_qmv_sm75.so"
                )
            var fwht_stream = UnsafePointer[Float32, MutAnyOrigin]()
            var fret = self.ffi_fwht_fn(
                self.x_dev, self.x_dev, self.signs_dev,
                Int32(M), Int32(self.K), Int32(0), fwht_stream
            )
            if fret != 0:
                x16h.free()
                raise Error(
                    "FWHT forward gagal: ret=" + String(Int(fret))
                    + " (N=" + String(self.N) + " K=" + String(self.K) + ")"
                )

        var ffi_success = False
        if self.ffi_ready:
            var cuda_stream = UnsafePointer[Float32, MutAnyOrigin]()
            var splits = decode_split_plan(self.N_packed, 1, self.K // 128)
            var ret = self.ffi_fn(
                self.x_dev, self.w_dev, self.s_dev, self.y_dev, self.ws_dev,
                Int32(M), Int32(self.N_packed), Int32(self.K), Int32(1),
                Int32(1), Int32(splits), cuda_stream
            )
            ffi_success = (ret == 0)
            if not ffi_success and self.bits != 1:
                x16h.free()
                raise Error(
                    "kernel GEMV 2-bit gagal: ret=" + String(Int(ret))
                    + " (N=" + String(self.N) + " K=" + String(self.K) + ")"
                )

        if not ffi_success:
            if self.bits != 1:
                x16h.free()
                raise Error(
                    "bits=2 memerlukan libbonsai_qmv_sm75.so dengan simbol " +
                    "launch_qmv_sm75_b2_decode_fp16 (FFI tidak aktif)"
                )
            qmv_sm75_1bit_launch_on[T](
                ctx,
                self.x_dev, self.w_dev, self.s_dev, self.y_dev,
                M, self.N_packed, self.K, 1, True, self.ws_dev
            )
        ctx.synchronize()
        var t3 = monotonic() if do_prof else 0
        var y16h = alloc[Scalar[T]](ny)
        ctx.enqueue_copy(y16h, self.y_dev_buf[])
        ctx.synchronize()
        var t4 = monotonic() if do_prof else 0
        for i in range(ny):
            out_ptr[i] = y16h[i].cast[DType.float32]()

        # Ekor dense (in_proj_b/a): hitung di host dari x ASLI (host F32,
        # tak ter-transform Hadamard — modul dense tidak terfold). Dilakukan
        # SETELAH salin kembali agar tidak tertimpa. n_tail==0 untuk bits=1
        # sehingga blok ini no-op pada jalur lama.
        if self.n_tail > 0:
            for row in range(M):
                for ti in range(self.n_tail):
                    var acc: Float32 = 0.0
                    var wbase = ti * self.K
                    var xbase = row * self.K
                    for ki in range(self.K):
                        acc += self.tail_w[wbase + ki] * x[xbase + ki]
                    out_ptr[row * self.N + self.N_packed + ti] = acc

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
