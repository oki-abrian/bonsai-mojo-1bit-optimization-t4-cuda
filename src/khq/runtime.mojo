# ===----------------------------------------------------------------------=== #
# Module: src/khq/runtime.mojo — runtime KV terkompresi KHQ (jalur decode).
# ===----------------------------------------------------------------------=== #
# Alur per langkah decode (satu token), per layer attention:
#   1. K post-norm PRE-rope ditulis ke ring unroped (kompresi tidak perlu
#      unrotate — inilah alasan K disimpan sebelum RoPE).
#      K yang sudah rope + V ditulis ke ring raw utk jendela presisi penuh.
#   2. Bila jendela mentok 256 slot -> kompres 128 slot tertua (watermark
#      256/128 ala referensi V21.23) lewat kernel CUDA khq_compress.
#   3. Attention = region terkompresi (kernel khq_attn, unnormalized) XOR
#      jendela raw (kernel khq_window_attn, unnormalized) -> merge logsumexp
#      + sigmoid gate (kernel khq_merge_gate).
#
# Format file centroid (ditulis src/khq/calib.mojo, magic 'KHQK' v2) — lihat
# komentar struct KhqCtx di bawah. Semua tensor centroid disimpan fp16.
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc
from gpu.host import DeviceBuffer
from math import sqrt
from io.file import FileHandle, open
from src.ops import (
    copy_vec_sm75_launch_on,
    khq_state_slot_cell,
    khq_compress_sm75_try_launch,
    khq_attn_sm75_try_launch,
    khq_window_attn_sm75_try_launch,
    khq_gather_q_sm75_try_launch,
    khq_merge_gate_try,
)
from os import getenv
from src.models.qwen3_5.linear import DeviceContextGPU

alias KHQ_MAX_LAYERS = 64
alias KHQ_RING = 256
alias KHQ_WATERMARK = 256
alias KHQ_CHUNK = 128
alias KHQ_MAGIC = 0x4B51484B  # 'KHQK'

# --------------------------------------------------------------- alokasi helper
fn _mk_f16(mut ctx: DeviceContextGPU, n: Int) raises -> UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]:
    var h = alloc[DeviceBuffer[DType.float16]](1)
    h.init_pointee_move(ctx.enqueue_create_buffer[DType.float16](n))
    return h

fn _mk_f32(mut ctx: DeviceContextGPU, n: Int) raises -> UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]:
    var h = alloc[DeviceBuffer[DType.float32]](1)
    h.init_pointee_move(ctx.enqueue_create_buffer[DType.float32](n))
    return h

fn _mk_u8(mut ctx: DeviceContextGPU, n: Int) raises -> UnsafePointer[DeviceBuffer[DType.uint8], MutAnyOrigin]:
    var h = alloc[DeviceBuffer[DType.uint8]](1)
    h.init_pointee_move(ctx.enqueue_create_buffer[DType.uint8](n))
    return h

fn _mk_u32(mut ctx: DeviceContextGPU, n: Int) raises -> UnsafePointer[DeviceBuffer[DType.uint32], MutAnyOrigin]:
    var h = alloc[DeviceBuffer[DType.uint32]](1)
    h.init_pointee_move(ctx.enqueue_create_buffer[DType.uint32](n))
    return h

# ------------------------------------------------------------------- struct
struct KhqLayer:
    """State KHQ satu layer attention (semua di VRAM)."""
    var ready: Bool
    var ts_k: Float32
    var alpha_k: Float32
    var ts_v: Float32
    var alpha_v: Float32
    var H_q: Int
    var H_kv: Int
    var D: Int
    var cap: Int
    var boundary: Int
    var write_pos: Int

    # tensor centroid (per layer)
    var h_dk: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_ck: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_rk: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_dv: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_cv: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_rv: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_vq: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]

    # ring raw (K roped, V) + ring K unroped (sumber kompresi)
    var h_ring_k: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_ring_v: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_ring_u: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]

    # buffer payload terkompresi
    var h_kmask: UnsafePointer[DeviceBuffer[DType.uint32], MutAnyOrigin]
    var h_kpay: UnsafePointer[DeviceBuffer[DType.uint8], MutAnyOrigin]
    var h_knorm: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_krn: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_vpay: UnsafePointer[DeviceBuffer[DType.uint8], MutAnyOrigin]
    var h_vnorm: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_vrn: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_vsm: UnsafePointer[DeviceBuffer[DType.uint8], MutAnyOrigin]

    # scratch
    var h_ktmp: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_qbuf: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var h_ac: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]
    var h_aw: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]

    fn __init__(out self):
        self.ready = False
        self.ts_k = 1.0
        self.alpha_k = 1.0
        self.ts_v = 1.95
        self.alpha_v = 1.0
        self.H_q = 0
        self.H_kv = 0
        self.D = 256
        self.cap = 0
        self.boundary = 0
        self.write_pos = 0
        self.h_dk = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_ck = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_rk = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_dv = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_cv = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_rv = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_vq = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_ring_k = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_ring_v = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_ring_u = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_kmask = UnsafePointer[DeviceBuffer[DType.uint32], MutAnyOrigin]()
        self.h_kpay = UnsafePointer[DeviceBuffer[DType.uint8], MutAnyOrigin]()
        self.h_knorm = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_krn = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_vpay = UnsafePointer[DeviceBuffer[DType.uint8], MutAnyOrigin]()
        self.h_vnorm = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_vrn = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_vsm = UnsafePointer[DeviceBuffer[DType.uint8], MutAnyOrigin]()
        self.h_ktmp = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_qbuf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.h_ac = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.h_aw = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()


struct KhqGlobals:
    var layers: UnsafePointer[KhqLayer, MutAnyOrigin]
    var n_layers: Int
    var active: Bool
    var ready: Bool
    var file_buf: UnsafePointer[UInt8, MutAnyOrigin]

    fn __init__(out self):
        self.layers = UnsafePointer[KhqLayer, MutAnyOrigin]()
        self.n_layers = 0
        self.active = False
        self.ready = False
        self.file_buf = UnsafePointer[UInt8, MutAnyOrigin]()


fn _g() -> UnsafePointer[KhqGlobals, MutAnyOrigin]:
    """State modul via slot statik lib CUDA (Mojo tanpa global var)."""
    var cell = khq_state_slot_cell(0).bitcast[UnsafePointer[KhqGlobals, MutAnyOrigin]]()
    var p = cell[]
    if p == UnsafePointer[KhqGlobals, MutAnyOrigin]():
        p = alloc[KhqGlobals](1)
        p[] = KhqGlobals()
        cell[] = p
    return p

fn khq_active() -> Bool:
    return _g()[].active


fn khq_activate(mut ctx: DeviceContextGPU, centroid_path: String, max_seq: Int) raises -> Bool:
    """Aktifkan jalur KHQ (dipanggil main saat env BONSAI_KHQ_PATH diset)."""
    if not khq_init(ctx, centroid_path, max_seq):
        return False
    _g()[].active = True
    print(">> [KHQ] jalur KV terkompresi AKTIF (watermark", KHQ_WATERMARK,
          "/", KHQ_CHUNK, ")")
    return True


fn _u32(p: UnsafePointer[UInt8, MutAnyOrigin], off: Int) -> UInt32:
    return UInt32(p[off]) | (UInt32(p[off + 1]) << 8) \
        | (UInt32(p[off + 2]) << 16) | (UInt32(p[off + 3]) << 24)


fn _f32(p: UnsafePointer[UInt8, MutAnyOrigin], off: Int) -> Float32:
    return (p + off).bitcast[Float32]()[]


fn _read_file(path: String, mut out_len: Int) raises -> UnsafePointer[UInt8, MutAnyOrigin]:
    var f = open(path, "r")
    var cap = 1 << 20
    var buf = alloc[UInt8](cap)
    var total = 0
    while True:
        if total == cap:
            var nb = alloc[UInt8](cap * 2)
            for i in range(cap):
                nb[i] = buf[i]
            buf.free()
            buf = nb
            cap *= 2
        var n = f.read(Span[UInt8, MutAnyOrigin](ptr=buf + total, length=cap - total))
        if n <= 0:
            break
        total += Int(n)
    f.close()
    out_len = total
    return buf


fn _upload_f16(
    mut ctx: DeviceContextGPU,
    holder: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin],
    src: UnsafePointer[UInt8, MutAnyOrigin],   # f32 host
    n: Int
) raises:
    var h = alloc[Scalar[DType.float16]](n)
    for i in range(n):
        h[i] = Scalar[DType.float16]((src + i * 4).bitcast[Float32]()[])
    ctx.enqueue_copy(holder[], h)
    h.free()


fn khq_init(mut ctx: DeviceContextGPU, path: String, max_seq: Int) raises -> Bool:
    """Muat file centroid + alokasi buffer. Sekali saja (idempoten)."""
    if _g()[].ready:
        return True
    var nbytes = 0
    _g()[].file_buf = _read_file(path, nbytes)
    if nbytes < 16 or _u32(_g()[].file_buf, 0) != KHQ_MAGIC:
        print(">> [KHQ] file centroid tidak valid:", path)
        return False
    var ver = Int(_u32(_g()[].file_buf, 4))
    var n_lay = Int(_u32(_g()[].file_buf, 8))
    if ver != 2:
        print(">> [KHQ] versi file tidak didukung:", ver)
        return False
    _g()[].layers = alloc[KhqLayer](KHQ_MAX_LAYERS)
    for i in range(KHQ_MAX_LAYERS):
        _g()[].layers[i] = KhqLayer()
    _g()[].n_layers = n_lay

    var off = 16
    for li in range(n_lay):
        var layer_id = Int(_u32(_g()[].file_buf, off))
        off += 4
        if layer_id < 0 or layer_id >= KHQ_MAX_LAYERS:
            print(">> [KHQ] layer_id di luar rentang:", layer_id)
            return False
        var L = UnsafePointer[KhqLayer, MutAnyOrigin](_g()[].layers + layer_id)

        # ---- K ----
        L[].ts_k = _f32(_g()[].file_buf, off)
        off += 4
        L[].alpha_k = _f32(_g()[].file_buf, off)
        off += 4
        L[].h_dk = _mk_f16(ctx, 256)
        _upload_f16(ctx, L[].h_dk, _g()[].file_buf + off, 256)
        off += 256 * 4
        L[].h_ck = _mk_f16(ctx, 4 * 256)
        _upload_f16(ctx, L[].h_ck, _g()[].file_buf + off, 4 * 256)
        off += 4 * 256 * 4
        L[].h_rk = _mk_f16(ctx, 64 * 16)
        _upload_f16(ctx, L[].h_rk, _g()[].file_buf + off, 64 * 16)
        off += 64 * 16 * 4

        # ---- V ----
        L[].ts_v = _f32(_g()[].file_buf, off)
        off += 4
        L[].alpha_v = _f32(_g()[].file_buf, off)
        off += 4
        L[].h_dv = _mk_f16(ctx, 256)
        _upload_f16(ctx, L[].h_dv, _g()[].file_buf + off, 256)
        off += 256 * 4
        L[].h_cv = _mk_f16(ctx, 4 * 256)
        _upload_f16(ctx, L[].h_cv, _g()[].file_buf + off, 4 * 256)
        off += 4 * 256 * 4
        L[].h_rv = _mk_f16(ctx, 64 * 16)
        _upload_f16(ctx, L[].h_rv, _g()[].file_buf + off, 64 * 16)
        off += 64 * 16 * 4
        L[].h_vq = _mk_f16(ctx, 64 * 256 * 3)
        _upload_f16(ctx, L[].h_vq, _g()[].file_buf + off, 64 * 256 * 3)
        off += 64 * 256 * 3 * 4

        # ---- ring + payload (ukuran tergantung config, diisi di init_layer) ----
        L[].cap = max_seq
        L[].ready = False

    _g()[].ready = True
    print(">> [KHQ] centroid dimuat:", n_lay, "layer dari", path)
    return True


fn khq_init_layer(
    mut ctx: DeviceContextGPU, layer_idx: Int, H_q: Int, H_kv: Int, d: Int
) raises:
    """Alokasi ring/payload satu layer — dipanggil dari step attention."""
    if layer_idx < 0 or layer_idx >= KHQ_MAX_LAYERS:
        return
    var L = UnsafePointer[KhqLayer, MutAnyOrigin](_g()[].layers + layer_idx)
    if L[].ready:
        return
    var kv = H_kv * d
    var cap = L[].cap
    L[].H_q = H_q
    L[].H_kv = H_kv
    L[].D = d
    L[].boundary = 0
    L[].write_pos = 0
    L[].h_ring_k = _mk_f16(ctx, KHQ_RING * kv)
    L[].h_ring_v = _mk_f16(ctx, KHQ_RING * kv)
    L[].h_ring_u = _mk_f16(ctx, KHQ_RING * kv)
    # payload disimpan token-major kontigu: baris = token*H_kv + head, jadi
    # basis baris = boundary*H_kv (bukan boundary saja). Ukuran buffer pun
    # cap*H_kv baris, bukan cap.
    var rows_total = cap * H_kv
    L[].h_kmask = _mk_u32(ctx, rows_total * 16)
    L[].h_kpay = _mk_u8(ctx, rows_total * 40)
    L[].h_knorm = _mk_f16(ctx, rows_total)
    L[].h_krn = _mk_f16(ctx, rows_total)
    L[].h_vpay = _mk_u8(ctx, rows_total * 104)
    L[].h_vnorm = _mk_f16(ctx, rows_total)
    L[].h_vrn = _mk_f16(ctx, rows_total)
    L[].h_vsm = _mk_u8(ctx, rows_total * 4)
    L[].h_ktmp = _mk_f16(ctx, kv)
    L[].h_qbuf = _mk_f16(ctx, H_q * d)
    L[].h_ac = _mk_f32(ctx, H_q * (d + 2))
    L[].h_aw = _mk_f32(ctx, H_q * (d + 2))
    L[].ready = True


fn khq_capture_unroped(
    mut ctx: DeviceContextGPU, layer_idx: Int,
    k_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
) raises:
    """Salin K post-norm PRE-rope ke scratch — dipanggil SEBELUM partial_rope
    supaya kompresi tidak perlu unrotate (kontrak user)."""
    if not _g()[].active or not _g()[].ready or layer_idx < 0 or layer_idx >= KHQ_MAX_LAYERS:
        return
    var L = UnsafePointer[KhqLayer, MutAnyOrigin](_g()[].layers + layer_idx)
    if not L[].ready:
        return
    copy_vec_sm75_launch_on[DType.float16](
        ctx, L[].h_ktmp[].unsafe_ptr(), k_dev, L[].H_kv * L[].D)


fn khq_step(
    mut ctx: DeviceContextGPU,
    layer_idx: Int,
    q_gate_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    k_roped_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    v_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    scale: Float32,
) raises -> Bool:
    """Satu langkah attention KHQ. False = jalur KHQ tidak tersedia/aktif."""
    if not _g()[].active or not _g()[].ready or layer_idx < 0 or layer_idx >= KHQ_MAX_LAYERS:
        return False
    var L = UnsafePointer[KhqLayer, MutAnyOrigin](_g()[].layers + layer_idx)
    if not L[].ready:
        return False
    var H_q = L[].H_q
    var H_kv = L[].H_kv
    var d = L[].D
    var kv = H_kv * d
    var slot = L[].write_pos % KHQ_RING

    # 1. tulis ke ring (K unroped utk kompresi; K roped + V utk jendela)
    copy_vec_sm75_launch_on[DType.float16](
        ctx, L[].h_ring_u[].unsafe_ptr() + slot * kv, L[].h_ktmp[].unsafe_ptr(), kv)
    copy_vec_sm75_launch_on[DType.float16](
        ctx, L[].h_ring_k[].unsafe_ptr() + slot * kv, k_roped_dev, kv)
    copy_vec_sm75_launch_on[DType.float16](
        ctx, L[].h_ring_v[].unsafe_ptr() + slot * kv, v_dev, kv)
    L[].write_pos += 1

    # 2. event kompresi (watermark 256 -> kompres 128 tertua, kontigu di ring)
    var window = L[].write_pos - L[].boundary
    if window >= KHQ_WATERMARK and L[].boundary + KHQ_CHUNK <= L[].cap:
        var s0 = L[].boundary % KHQ_RING
        var nrows = KHQ_CHUNK * H_kv
        var r0 = L[].boundary * H_kv
        var uptr = L[].h_ring_u[].unsafe_ptr() + s0 * kv
        var vptr = L[].h_ring_v[].unsafe_ptr() + s0 * kv
        var ok_k = khq_compress_sm75_try_launch(
            uptr, L[].h_dk[].unsafe_ptr(), L[].h_rk[].unsafe_ptr(),
            L[].h_ck[].unsafe_ptr(), L[].h_ck[].unsafe_ptr(),
            L[].h_kmask[].unsafe_ptr() + r0 * 16,
            L[].h_knorm[].unsafe_ptr() + r0,
            L[].h_krn[].unsafe_ptr() + r0,
            L[].h_kpay[].unsafe_ptr() + r0 * 40,
            L[].h_vsm[].unsafe_ptr() + r0 * 4,
            nrows, d, L[].ts_k, L[].alpha_k, False)
        var ok_v = khq_compress_sm75_try_launch(
            vptr, L[].h_dv[].unsafe_ptr(), L[].h_rv[].unsafe_ptr(),
            L[].h_cv[].unsafe_ptr(), L[].h_vq[].unsafe_ptr(),
            L[].h_kmask[].unsafe_ptr() + r0 * 16,
            L[].h_vnorm[].unsafe_ptr() + r0,
            L[].h_vrn[].unsafe_ptr() + r0,
            L[].h_vpay[].unsafe_ptr() + r0 * 104,
            L[].h_vsm[].unsafe_ptr() + r0 * 4,
            nrows, d, L[].ts_v, L[].alpha_v, True)
        if not (ok_k and ok_v):
            return False
        # Bukti cadence + ISI (BONSAI_KHQ_DEBUG=1): tiap event mencetak boundary,
        # jumlah event, sisa jendela raw, dan norma tiap baris yang baru
        # dikompres (bit fp16, integer — eksak & mudah diparse). Norma ini
        # fingerprint isi: bila baris r berisi token yang benar, normanya harus
        # sama dengan norma K token itu di dump asli.
        var dbg = getenv("BONSAI_KHQ_DEBUG")
        if dbg and dbg[0] == "1":
            var nrm = alloc[Scalar[DType.float16]](L[].cap * H_kv)
            ctx.enqueue_copy(nrm, L[].h_knorm[])
            ctx.synchronize()
            var bits = nrm.bitcast[UInt16]()
            var line = String(">> [KHQ-NORM] ") + String(layer_idx) + " " \
                + String(L[].boundary) + " " + String(nrows) + " "
            for i in range(nrows):
                line += String(Int(bits[r0 + i])) + " "
            print(line)
            nrm.free()
        L[].boundary += KHQ_CHUNK
        if dbg and dbg[0] == "1":
            print(">> [KHQ-COMPRESS] layer", layer_idx, "event#",
                  L[].boundary // KHQ_CHUNK, "boundary", L[].boundary,
                  "write_pos", L[].write_pos,
                  "raw_window", L[].write_pos - L[].boundary)

    # 3. attention: Q dipisah dari buffer interleaved (dipakai kedua region)
    var c_len = L[].boundary
    var has_c = c_len > 0
    var win_len = L[].write_pos - c_len
    var has_w = win_len > 0
    if not (has_c or has_w):
        return False
    if not khq_gather_q_sm75_try_launch(
        q_gate_dev, L[].h_qbuf[].unsafe_ptr(), H_q, d):
        return False

    if has_c:
        var n_rep = H_q // H_kv
        var ok = khq_attn_sm75_try_launch(
            L[].h_qbuf[].unsafe_ptr(),
            L[].h_kpay[].unsafe_ptr(), L[].h_vpay[].unsafe_ptr(),
            L[].h_vsm[].unsafe_ptr(), L[].h_kmask[].unsafe_ptr(),
            L[].h_knorm[].unsafe_ptr(), L[].h_krn[].unsafe_ptr(),
            L[].h_vnorm[].unsafe_ptr(), L[].h_vrn[].unsafe_ptr(),
            L[].h_ck[].unsafe_ptr(), L[].h_cv[].unsafe_ptr(),
            L[].h_vq[].unsafe_ptr(), L[].h_rk[].unsafe_ptr(),
            L[].h_rv[].unsafe_ptr(), L[].h_dk[].unsafe_ptr(),
            L[].h_dv[].unsafe_ptr(), L[].h_ac[].unsafe_ptr(),
            H_q, H_q, 1, c_len, c_len, n_rep, scale,
            L[].alpha_k, L[].alpha_v, 10000000.0, 0)
        if not ok:
            return False
        # jendela raw: bagian KROPE yang belum dikompresi (dari ujung ring);
        # posisi absolut token terakhir = write_pos-1, jadi start_pos RoPE
        # efektif untuk kernel kompresi di atas = c_len-1.

    if has_w:
        var win_start = c_len % KHQ_RING
        var ok = khq_window_attn_sm75_try_launch(
            L[].h_qbuf[].unsafe_ptr(), L[].h_ring_k[].unsafe_ptr(),
            L[].h_ring_v[].unsafe_ptr(), L[].h_aw[].unsafe_ptr(),
            win_start, win_len, KHQ_RING, H_q, H_kv, d, scale)
        if not ok:
            return False

    # 4. merge + sigmoid gate
    return khq_merge_gate_try(
        L[].h_ac[].unsafe_ptr(), L[].h_aw[].unsafe_ptr(), q_gate_dev,
        out_dev, H_q, d, has_w, has_c)
