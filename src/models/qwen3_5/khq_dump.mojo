# ===----------------------------------------------------------------------=== #
# Module: src/models/qwen3_5/khq_dump.mojo
# Purpose: Dump aktivasi K/V untuk kalibrasi KHQ (KudaHitamQuant).
# ===----------------------------------------------------------------------=== #
# K diambil SETELAH head-RMSNorm dan SEBELUM partial RoPE (post-norm, pre-rope):
# nilai unroped — kalibrasi tidak perlu unrotate lagi (kontrak referensi).
# V diambil apa adanya dari proyeksi (sebelum masuk KV cache).
#
# Aktif hanya bila env BONSAI_DUMP_KV_DIR diset (lihat main.mojo).
# Output satu file <dir>/kv_dump.bin (little-endian):
#   u32 magic 'KHQD' | u32 n_layers | u32 dim
#   per layer: u32 layer_id | u32 n_tokens   (header blok)
#   data   : per layer berurutan -> K (n*dim f32) lalu V (n*dim f32)
# Baris = token berurutan, lebar dim = H_kv*head_dg()[].dim (4*256 = 1024).

from memory import UnsafePointer, alloc
from gpu.host import DeviceBuffer
from math import exp
from .linear import DeviceContextGPU
from src.ops import copy_vec_sm75_launch_on, khq_gather_q_sm75_try_launch, khq_state_slot_cell
from io.file import FileHandle, open

alias KHQ_MAX_LAYERS = 64

# state modul dipindah ke KhqDumpGlobals (lihat _dg) — Mojo tanpa global var

fn khq_dump_configure(dir_path: String, max_tokens: Int, kvd: Int):
    _dg()[].enabled = True
    _dg()[].dpath = dir_path
    _dg()[].dim = kvd
    _dg()[].cap = max_tokens
    _dg()[].n = alloc[Int](KHQ_MAX_LAYERS)
    _dg()[].kbuf = alloc[UnsafePointer[Float32, MutAnyOrigin]](KHQ_MAX_LAYERS)
    _dg()[].vbuf = alloc[UnsafePointer[Float32, MutAnyOrigin]](KHQ_MAX_LAYERS)
    for i in range(KHQ_MAX_LAYERS):
        _dg()[].n[i] = 0
        _dg()[].kbuf[i] = UnsafePointer[Float32, MutAnyOrigin]()
        _dg()[].vbuf[i] = UnsafePointer[Float32, MutAnyOrigin]()

fn khq_dump_active() -> Bool:
    return _dg()[].enabled


fn khq_dump_kv(
    mut ctx: DeviceContextGPU,
    layer_idx: Int,
    k_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    v_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
) raises:
    """Salin K/V satu token dari VRAM ke buffer host (dipanggil setelah
    head-RMSNorm K, sebelum RoPE)."""
    if not _dg()[].enabled or layer_idx < 0 or layer_idx >= KHQ_MAX_LAYERS:
        return
    var n = _dg()[].n[layer_idx]
    if n >= _dg()[].cap:
        return
    var dim = _dg()[].dim

    # Staging device + host: dialokasi sekali, dipakai ulang tiap token.
    if not _dg()[].ready:
        _dg()[].stage_k = alloc[DeviceBuffer[DType.float16]](1)
        _dg()[].stage_k.init_pointee_move(ctx.enqueue_create_buffer[DType.float16](dim))
        _dg()[].stage_v = alloc[DeviceBuffer[DType.float16]](1)
        _dg()[].stage_v.init_pointee_move(ctx.enqueue_create_buffer[DType.float16](dim))
        _dg()[].host = alloc[Scalar[DType.float16]](2 * dim)
        _dg()[].ready = True

    copy_vec_sm75_launch_on[DType.float16](ctx, _dg()[].stage_k[].unsafe_ptr(), k_dev, dim)
    copy_vec_sm75_launch_on[DType.float16](ctx, _dg()[].stage_v[].unsafe_ptr(), v_dev, dim)
    ctx.synchronize()
    ctx.enqueue_copy(_dg()[].host, _dg()[].stage_k[])
    ctx.enqueue_copy(_dg()[].host + dim, _dg()[].stage_v[])
    ctx.synchronize()

    if _dg()[].kbuf[layer_idx] == UnsafePointer[Float32, MutAnyOrigin]():
        _dg()[].kbuf[layer_idx] = alloc[Float32](_dg()[].cap * dim)
        _dg()[].vbuf[layer_idx] = alloc[Float32](_dg()[].cap * dim)
    for i in range(dim):
        _dg()[].kbuf[layer_idx][n * dim + i] = Float32(_dg()[].host[i])
        _dg()[].vbuf[layer_idx][n * dim + i] = Float32(_dg()[].host[dim + i])
    _dg()[].n[layer_idx] = n + 1


fn _pack_u32(p: UnsafePointer[UInt8, MutAnyOrigin], v: UInt32):
    p[0] = UInt8(v & 0xFF)
    p[1] = UInt8((v >> 8) & 0xFF)
    p[2] = UInt8((v >> 16) & 0xFF)
    p[3] = UInt8((v >> 24) & 0xFF)


# --------------------------------------------------------------- dump attention
# Skor softmax (post-causal) untuk kalibrasi SmartVQ (attention-aware loss),
# referensi precompute_centroids.py: collected_attn = softmax(qk/sqrt(d)+mask).
# K & Q yang dipakai SUDAH di-rope (posisi absolut), layaknya skor asli.
# File per layer <dir>/attn_<layer>.bin:
#   u32 'KHQA' | u32 H_q | u32 H_kv | u32 D
#   per token: u32 n_valid (=pos+1) lalu H_q*n_valid skor fp16 (baris q-head).
fn f32_to_half_bits(sp: UnsafePointer[Float32, MutAnyOrigin], v: Float32) -> UInt16:
    # IEEE half round-to-nearest-even (bit pattern fp16)
    sp[0] = v
    var bits = sp.bitcast[UInt32]()[]
    var sign = UInt16((bits >> 16) & 0x8000)
    var ex = Int((bits >> 23) & 0xFF) - 127 + 15
    var man = bits & 0x7FFFFF
    if ex >= 31:
        return sign | 0x7C00
    if ex > 0:
        var half = (UInt16(ex) << 10) | UInt16(man >> 13)
        var rem = man & 0x1FFF
        if rem > 0x1000 or (rem == 0x1000 and (half & 1) == 1):
            half += 1
        return sign | half
    if ex < -9:
        return sign
    # subnormal fp16: man16 = round((512 + m/16384) * 2^exp)
    var val16 = (512.0 + Float32(man) / 16384.0) * exp(Float32(ex) * 0.6931471805599453)
    return sign | UInt16(val16 + 0.5)


struct _AttnDump:
    var q_stage: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var k_stage: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var host: UnsafePointer[Scalar[DType.float16], MutAnyOrigin]
    var krolls: UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin]  # per layer (cap, H_kv*D) roped K
    var logits: UnsafePointer[Float32, MutAnyOrigin]  # (H_q, cap)
    var bytes_buf: UnsafePointer[UInt8, MutAnyOrigin] # (H_q*cap*2)
    var scratch: UnsafePointer[Float32, MutAnyOrigin] # 1 elemen: reinterpret f32 -> bit
    var files: UnsafePointer[FileHandle, MutAnyOrigin]
    var opened: UnsafePointer[Bool, MutAnyOrigin]
    var ready: Bool
    var H_q: Int
    var H_kv: Int
    var D: Int

    fn __init__(out self):
        self.q_stage = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.k_stage = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.host = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.krolls = UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin]()
        self.logits = UnsafePointer[Float32, MutAnyOrigin]()
        self.bytes_buf = UnsafePointer[UInt8, MutAnyOrigin]()
        self.scratch = UnsafePointer[Float32, MutAnyOrigin]()
        self.files = UnsafePointer[FileHandle, MutAnyOrigin]()
        self.opened = UnsafePointer[Bool, MutAnyOrigin]()
        self.ready = False
        self.H_q = 0
        self.H_kv = 0
        self.D = 0


struct KhqDumpGlobals:
    var enabled: Bool
    var ready: Bool
    var dpath: String
    var dim: Int
    var cap: Int
    var n: UnsafePointer[Int, MutAnyOrigin]
    var kbuf: UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin]
    var vbuf: UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin]
    var stage_k: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var stage_v: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var host: UnsafePointer[Scalar[DType.float16], MutAnyOrigin]
    var ad: _AttnDump

    fn __init__(out self):
        self.enabled = False
        self.ready = False
        self.dpath = String()
        self.dim = 0
        self.cap = 0
        self.n = UnsafePointer[Int, MutAnyOrigin]()
        self.kbuf = UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin]()
        self.vbuf = UnsafePointer[UnsafePointer[Float32, MutAnyOrigin], MutAnyOrigin]()
        self.stage_k = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.stage_v = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.host = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.ad = _AttnDump()


fn _dg() -> UnsafePointer[KhqDumpGlobals, MutAnyOrigin]:
    """State modul via slot statik lib CUDA (slot 1)."""
    var cell = khq_state_slot_cell(1).bitcast[UnsafePointer[KhqDumpGlobals, MutAnyOrigin]]()
    var p = cell[]
    if p == UnsafePointer[KhqDumpGlobals, MutAnyOrigin]():
        p = alloc[KhqDumpGlobals](1)
        p[] = KhqDumpGlobals()
        cell[] = p
    return p


fn khq_dump_attn(
    mut ctx: DeviceContextGPU,
    layer_idx: Int,
    q_gate_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],  # roped, interleaved
    k_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],       # roped
    pos: Int,
    H_q_in: Int, H_kv_in: Int, D_in: Int,
    scale: Float32,
) raises:
    """Rekam skor softmax attention token ini (dipanggil SETELAH RoPE)."""
    if not _dg()[].enabled or layer_idx < 0 or layer_idx >= KHQ_MAX_LAYERS:
        return
    if pos >= _dg()[].cap:
        return
    var H_q = H_q_in
    var H_kv = H_kv_in
    var dim = _dg()[].dim

    if not _dg()[].ad.ready:
        _dg()[].ad.H_q = H_q
        _dg()[].ad.H_kv = H_kv
        _dg()[].ad.D = D_in
        _dg()[].ad.q_stage = alloc[DeviceBuffer[DType.float16]](1)
        _dg()[].ad.q_stage.init_pointee_move(ctx.enqueue_create_buffer[DType.float16](H_q * D_in))
        _dg()[].ad.k_stage = alloc[DeviceBuffer[DType.float16]](1)
        _dg()[].ad.k_stage.init_pointee_move(ctx.enqueue_create_buffer[DType.float16](dim))
        _dg()[].ad.host = alloc[Scalar[DType.float16]](H_q * D_in + dim)
        _dg()[].ad.krolls = alloc[UnsafePointer[Float32, MutAnyOrigin]](KHQ_MAX_LAYERS)
        _dg()[].ad.logits = alloc[Float32](H_q * _dg()[].cap)
        _dg()[].ad.bytes_buf = alloc[UInt8](H_q * _dg()[].cap * 2)
        _dg()[].ad.scratch = alloc[Float32](1)
        _dg()[].ad.files = alloc[FileHandle](KHQ_MAX_LAYERS)
        _dg()[].ad.opened = alloc[Bool](KHQ_MAX_LAYERS)
        for i in range(KHQ_MAX_LAYERS):
            _dg()[].ad.opened[i] = False
            _dg()[].ad.krolls[i] = UnsafePointer[Float32, MutAnyOrigin]()
        _dg()[].ad.ready = True
    if H_q != _dg()[].ad.H_q or H_kv != _dg()[].ad.H_kv or D_in != _dg()[].ad.D:
        return

    # Buffer K-history PER LAYER. Loop decode bersifat layer-major, jadi satu
    # buffer bersama akan berisi K dari layer attention TERAKHIR, bukan layer
    # ini — skor attention (calon collected_attn) jadi salah.
    if _dg()[].ad.krolls[layer_idx] == UnsafePointer[Float32, MutAnyOrigin]():
        _dg()[].ad.krolls[layer_idx] = alloc[Float32](_dg()[].cap * dim)
    var kroll = _dg()[].ad.krolls[layer_idx]
    if pos == 0:
        for i in range(_dg()[].cap * dim):
            kroll[i] = 0.0

    if not _dg()[].ad.opened[layer_idx]:
        var path = _dg()[].dpath + "/attn_" + String(layer_idx) + ".bin"
        var f = open(path, "w")
        var hdr = alloc[UInt8](16)
        _pack_u32(hdr, 0x4151484B)  # 'KHQA'
        _pack_u32(hdr + 4, UInt32(H_q))
        _pack_u32(hdr + 8, UInt32(H_kv))
        _pack_u32(hdr + 12, UInt32(D_in))
        _ = f.write_bytes(Span[UInt8, MutAnyOrigin](ptr=hdr, length=16))
        hdr.free()
        _dg()[].ad.files[layer_idx] = f ^
        _dg()[].ad.opened[layer_idx] = True

    # salin q (gather interleaved [H_q,2D] -> [H_q,D]) dan k roped
    if not khq_gather_q_sm75_try_launch(
        q_gate_dev, _dg()[].ad.q_stage[].unsafe_ptr(), H_q, D_in):
        raise Error("KHQ-DUMP: butuh libbonsai CUDA untuk gather Q")
    copy_vec_sm75_launch_on[DType.float16](
        ctx, _dg()[].ad.k_stage[].unsafe_ptr(), k_dev, dim)
    ctx.synchronize()
    ctx.enqueue_copy(_dg()[].ad.host, _dg()[].ad.q_stage[])
    ctx.enqueue_copy(_dg()[].ad.host + (H_q * D_in), _dg()[].ad.k_stage[])
    ctx.synchronize()

    # append K roped ke rolling buffer layer ini
    var kroll_pos = kroll + pos * dim
    for i in range(dim):
        kroll_pos[i] = Float32(_dg()[].ad.host[H_q * D_in + i])

    # skor: softmax atas s=0..pos (GQA: kv-head = h / n_rep)
    var n_rep = H_q // H_kv
    var n_valid = pos + 1
    for h in range(H_q):
        var kv = h // n_rep
        var qrow = _dg()[].ad.host + h * D_in  # q_gate interleaved: q di 0..D-1
        var mxv: Float32 = -1e30
        for s in range(n_valid):
            var acc: Float32 = 0.0
            var krow = kroll + s * dim + kv * D_in
            for d in range(D_in):
                acc += Float32(qrow[d]) * krow[d]
            acc *= scale
            _dg()[].ad.logits[h * _dg()[].cap + s] = acc
            if acc > mxv:
                mxv = acc
        var zsum: Float32 = 0.0
        for s in range(n_valid):
            var e = exp(_dg()[].ad.logits[h * _dg()[].cap + s] - mxv)
            _dg()[].ad.logits[h * _dg()[].cap + s] = e
            zsum += e
        var zinv = 1.0 / zsum
        var bp = _dg()[].ad.bytes_buf + h * n_valid * 2
        for s in range(n_valid):
            var hb = f32_to_half_bits(
                _dg()[].ad.scratch, _dg()[].ad.logits[h * _dg()[].cap + s] * zinv)
            bp[s * 2] = UInt8(hb & 0xFF)
            bp[s * 2 + 1] = UInt8(hb >> 8)

    var rec = alloc[UInt8](4 + H_q * n_valid * 2)
    _pack_u32(rec, UInt32(n_valid))
    for i in range(H_q * n_valid * 2):
        rec[4 + i] = _dg()[].ad.bytes_buf[i]
    _ = _dg()[].ad.files[layer_idx].write_bytes(
        Span[UInt8, MutAnyOrigin](ptr=rec, length=4 + H_q * n_valid * 2))
    rec.free()


fn khq_dump_flush() raises:
    """Tulis semua layer yang terisi ke <dir>/kv_dump.bin lalu bebankan."""
    if not _dg()[].enabled:
        return
    if _dg()[].ad.ready:
        for li in range(KHQ_MAX_LAYERS):
            if _dg()[].ad.opened[li]:
                _dg()[].ad.files[li].close()
                _dg()[].ad.opened[li] = False
    var n_layers = 0
    for li in range(KHQ_MAX_LAYERS):
        if _dg()[].n[li] > 0:
            n_layers += 1
    if n_layers == 0:
        print(">> [KHQ-DUMP] tidak ada data — dilewati")
        return

    var path = _dg()[].dpath + "/kv_dump.bin"
    var f = open(path, "w")
    var hdr = alloc[UInt8](12 + 8 * n_layers)
    _pack_u32(hdr, 0x4451484B)          # 'KHQD'
    _pack_u32(hdr + 4, UInt32(n_layers))
    _pack_u32(hdr + 8, UInt32(_dg()[].dim))
    var off = 12
    for li in range(KHQ_MAX_LAYERS):
        if _dg()[].n[li] == 0:
            continue
        _pack_u32(hdr + off, UInt32(li))
        _pack_u32(hdr + off + 4, UInt32(_dg()[].n[li]))
        off += 8
    _ = f.write_bytes(Span[UInt8, MutAnyOrigin](ptr=hdr, length=12 + 8 * n_layers))

    for li in range(KHQ_MAX_LAYERS):
        var n = _dg()[].n[li]
        if n == 0:
            continue
        var nb = n * _dg()[].dim * 4
        _ = f.write_bytes(Span[UInt8, MutAnyOrigin](
            ptr=_dg()[].kbuf[li].bitcast[UInt8](), length=nb))
        _ = f.write_bytes(Span[UInt8, MutAnyOrigin](
            ptr=_dg()[].vbuf[li].bitcast[UInt8](), length=nb))
        _dg()[].kbuf[li].free()
        _dg()[].vbuf[li].free()
        _dg()[].kbuf[li] = UnsafePointer[Float32, MutAnyOrigin]()
        _dg()[].vbuf[li] = UnsafePointer[Float32, MutAnyOrigin]()
        _dg()[].n[li] = 0
    f.close()
    hdr.free()
    print(">> [KHQ-DUMP] flushed", n_layers, "layer ->", path)
