# ===----------------------------------------------------------------------=== #
# Module: src/khq/calib.mojo — Kalibrasi centroid KHQ jalur K (port referensi).
# ===----------------------------------------------------------------------=== #
# Sumber (dibaca langsung, bukan asumsi):
#   KudaHitamQuant_full-reasoning.py: calibrate() jalur is_v=False (baris 457-485)
#     -> centroid [0, -c, +c, +c], c=0.9350*sigma, ts = 16/expected_norm,
#        cents *= ts   (faktor 16 menetralkan f_s=1/16 di kernel compress)
#   precompute_centroids.py: turnamen SmartK (baris 281-417)
#     -> grid 8x8 4 fase, 3 jalur MSE (flat/mag/var), juri max-error
#   KudaHitamMLX.py: CUDA_SOURCE jalur K (baris 576-835) + decompress_k
#     (KudaHitamQuant_full-reasoning.py baris 897-1012)
#
# Alur: baca dump K (post-norm, PRE-rope) -> norm+FWHT+rotor (fp32) ->
#       kalibrasi statis -> turnamen -> tulis file centroid.
# Jalankan: mojo run src/khq/calib.mojo <dir_dump> <out.bin>
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc
from math import sqrt, atan2, cos, sin, exp, log, floor
from io.file import FileHandle, open
from sys import argv
from os import getenv
from sys.ffi import OwnedDLHandle
from gpu.host import DeviceBuffer
from src.models.qwen3_5.linear import DeviceContextGPU
from src.ops import cal_svq_train_try_launch, try_open_cuda_lib

alias DevF32 = DeviceBuffer[DType.float32]


fn dev_f32(
    ctx: DeviceContextGPU, p: UnsafePointer[Float32, MutAnyOrigin], n: Int
) raises -> DevF32:
    """Upload host -> device (float32)."""
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(buf, p)
    ctx.synchronize()
    return buf


fn dev_new(ctx: DeviceContextGPU, n: Int) raises -> DevF32:
    return ctx.enqueue_create_buffer[DType.float32](n)


fn d2h_f32(
    ctx: DeviceContextGPU, buf: DevF32, p: UnsafePointer[Float32, MutAnyOrigin],
    n: Int
) raises:
    ctx.enqueue_copy(p, buf)
    ctx.synchronize()


alias CalOpFn = fn(
    Int32,
    UnsafePointer[Float32, MutAnyOrigin],  # a
    UnsafePointer[Float32, MutAnyOrigin],  # b
    UnsafePointer[Float32, MutAnyOrigin],  # c
    UnsafePointer[Float32, MutAnyOrigin],  # d
    UnsafePointer[Float32, MutAnyOrigin],  # e
    UnsafePointer[Float32, MutAnyOrigin],  # f
    UnsafePointer[Float32, MutAnyOrigin],  # g
    Float32, Float32,
    Int32, Int32, Int32, Int32,
    UnsafePointer[Float32, MutAnyOrigin]   # stream (null)
) -> Int32


# op: 1=FWHT-base 2=K-static 3=rotate 4=PCA 5=kmeans4 6=kmeans-VQ
#     7=Procrustes 8=SmartK 9=SmartV 10=head-major
fn cal_op(
    op: Int32,
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    c: UnsafePointer[Float32, MutAnyOrigin],
    d: UnsafePointer[Float32, MutAnyOrigin],
    e: UnsafePointer[Float32, MutAnyOrigin],
    f: UnsafePointer[Float32, MutAnyOrigin],
    g: UnsafePointer[Float32, MutAnyOrigin],
    f1: Float32, f2: Float32,
    i1: Int, i2: Int, i3: Int, i4: Int
) -> Bool:
    """Satu langkah kalibrasi non-SmartVQ di T4 via launch_cal_op."""
    var disable = getenv("BONSAI_DISABLE_CUDA_FFI")
    if disable and (disable == "1" or disable == "true"):
        return False
    try:
        var h = try_open_cuda_lib()
        var fn_h = h.get_function[CalOpFn]("launch_cal_op")
        var ret = fn_h(
            op, a, b, c, d, e, f, g, f1, f2,
            Int32(i1), Int32(i2), Int32(i3), Int32(i4),
            UnsafePointer[Float32, MutAnyOrigin]()
        )
        return ret == 0
    except:
        return False


alias D = 256                 # head_dim (kontrak referensi)
alias NP = 64                 # D // 4
alias OUT_MAX = 50            # kapasitas outlier payload K
alias FS = 0.0625             # 1/sqrt(256) — f_s kernel
alias TOURN_ROWS = 4096       # subset turnamen (referensi: all_k[:4096])
alias MAGIC_KHQK = 0x4B51484B # 'KHQK'
alias MAGIC_KHQD = 0x4451484B # 'KHQD' (dump)

# --------------------------------------------------------------- util dasar
fn fabs(x: Float32) -> Float32:
    return x if x >= 0.0 else -x

fn fp16(x: Float32) -> Float32:
    # nilai dibulatkan ke grid fp16 (kernel menerima half; math di fp32)
    var h = Scalar[DType.float16](x)
    return Float32(h)

fn round_half_away(x: Float32) -> Float32:
    # roundf C: setengah menjauhi nol
    if x >= 0.0:
        return Float32(Int(x + 0.5))
    return -Float32(Int(-x + 0.5))

fn pack_u32(p: UnsafePointer[UInt8, MutAnyOrigin], off: Int, v: UInt32):
    p[off + 0] = UInt8(v & 0xFF)
    p[off + 1] = UInt8((v >> 8) & 0xFF)
    p[off + 2] = UInt8((v >> 16) & 0xFF)
    p[off + 3] = UInt8((v >> 24) & 0xFF)

fn unpack_u32(p: UnsafePointer[UInt8, MutAnyOrigin], off: Int) -> UInt32:
    return UInt32(p[off + 0]) | (UInt32(p[off + 1]) << 8) \
        | (UInt32(p[off + 2]) << 16) | (UInt32(p[off + 3]) << 24)

fn write_f32(mut f: FileHandle, buf: UnsafePointer[Float32, MutAnyOrigin], n: Int) raises:
    _ = f.write_bytes(Span[UInt8, MutAnyOrigin](
        ptr=buf.bitcast[UInt8](), length=n * 4))

# ---------------------------------------------------- d vector {-1,+1} per layer
fn make_d(k: UnsafePointer[Float32, MutAnyOrigin], layer_id: Int):
    var s = UInt32(layer_id) * 2654435761 + 12345
    for i in range(D):
        s ^= s << 13
        s ^= s >> 17
        s ^= s << 5
        k[i] = Float32(1.0) if (s & 1) == 1 else Float32(-1.0)

# ============================== SMARTVQ (GPU) ==============================
# Port train_smartvq.py — SEMUA di T4 via launch_cal_svq_train (qmv_sm75_kernel.cu):
# VQ-STE, W_trm rank-8 (I + A@B), rotor beku, Adam, temp 1.5->0.1, patience 15,
# warm-restart, turnamen seed [42,137,271] ronde 30+70 — semua kernel CUDA.
# Mojo hanya: upload data, satu panggilan FFI, ambil codebook winner.

fn half_bits_to_f32(h: UInt16) -> Float32:
    var neg = (h >> 15) == 1
    var e = Int((h >> 10) & 0x1F)
    var m = Int(h & 0x3FF)
    var out: Float32
    if e == 0:
        out = Float32(m) * 5.9604644775390625e-08
    elif e == 31:
        out = 1e30
    else:
        out = (Float32(m) / 1024.0 + 1.0) * exp(Float32(e - 15) * 0.6931471805599453)
    return -out if neg else out


fn svq_gpu_run(
    ctx: DeviceContextGPU,
    v_dev: UnsafePointer[Float32, MutAnyOrigin],   # device (rows,D) head-major
    rows: Int,
    T: Int, H_q: Int, H_kv: Int,
    attn_host: UnsafePointer[Float32, MutAnyOrigin],
    vq_dev: UnsafePointer[Float32, MutAnyOrigin],  # device in/out (NP*768)
    rp_dev: UnsafePointer[Float32, MutAnyOrigin]   # device (NP*16)
) raises -> Bool:
    """SmartVQ via pointer device: hanya attn yang di-upload; codebook
    winner ditulis balik ke vq_dev (out = in, aman urutan stream)."""
    var n_seeds = 1
    if attn_host:
        n_seeds = 3
    var b_attn = ctx.enqueue_create_buffer[DType.float32](1)
    var attn_ptr = UnsafePointer[Float32, MutAnyOrigin]()
    if attn_host:
        b_attn = ctx.enqueue_create_buffer[DType.float32](H_q * T * T)
        ctx.enqueue_copy(b_attn, attn_host)
        attn_ptr = b_attn.unsafe_ptr()
    ctx.synchronize()
    var ok = cal_svq_train_try_launch(
        v_dev, attn_ptr, vq_dev, rp_dev, vq_dev,
        UnsafePointer[Float32, MutAnyOrigin](),
        rows, T, H_q, H_kv, 100, n_seeds, 0.002)
    ctx.synchronize()
    return ok


# ============================================================================ #

# ------------------------------------------------------- vektor d untuk V
fn make_d_v(k: UnsafePointer[Float32, MutAnyOrigin], layer_id: Int):
    var s = UInt32(layer_id) * 40503 + 987654321
    for i in range(D):
        s ^= s << 13
        s ^= s >> 17
        s ^= s << 5
        k[i] = Float32(1.0) if (s & 1) == 1 else Float32(-1.0)

# (kalibrasi V penuh di GPU — launch_cal_op 1..10)


# ------------------------------------------------------------------ driver
fn slurp(
    path: String, out_len: UnsafePointer[Int, MutAnyOrigin]
) raises -> UnsafePointer[UInt8, MutAnyOrigin]:
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
    out_len[0] = total
    return buf


fn load_attn_layer(
    dir_path: String, layer_id: Int,
    out_H_q: UnsafePointer[Int, MutAnyOrigin],
    out_T: UnsafePointer[Int, MutAnyOrigin]
) -> UnsafePointer[Float32, MutAnyOrigin]:
    """Baca <dir>/attn_<layer>.bin ('KHQA') -> (H_q, T, T) f32. Null bila absen."""
    var path = dir_path + "/attn_" + String(layer_id) + ".bin"
    var nbytes_p = alloc[Int](1)
    nbytes_p[0] = 0
    var buf = UnsafePointer[UInt8, MutAnyOrigin]()
    try:
        buf = slurp(path, nbytes_p)
    except:
        nbytes_p.free()
        return UnsafePointer[Float32, MutAnyOrigin]()
    var nbytes = nbytes_p[0]
    nbytes_p.free()
    if nbytes < 16 or unpack_u32(buf, 0) != 0x4151484B:
        buf.free()
        return UnsafePointer[Float32, MutAnyOrigin]()
    var hq = Int(unpack_u32(buf, 4))
    var tkv = Int(unpack_u32(buf, 8))
    var dd = Int(unpack_u32(buf, 12))
    _ = tkv
    _ = dd
    var off = 16
    var cnt = 0
    while off + 4 <= nbytes:
        var nv = Int(unpack_u32(buf, off))
        off += 4 + hq * nv * 2
        cnt += 1
    if cnt == 0:
        buf.free()
        return UnsafePointer[Float32, MutAnyOrigin]()
    var out = alloc[Float32](hq * cnt * cnt)
    for i in range(hq * cnt * cnt):
        out[i] = 0.0
    off = 16
    for t in range(cnt):
        var nv = Int(unpack_u32(buf, off))
        off += 4
        for h in range(hq):
            for s in range(nv):
                var b0 = UInt16(buf[off]) | (UInt16(buf[off + 1]) << 8)
                out[(h * cnt + t) * cnt + s] = half_bits_to_f32(b0)
                off += 2
    buf.free()
    out_H_q[0] = hq
    out_T[0] = cnt
    return out


fn main() raises:
    var args = argv()
    if len(args) < 3:
        print("pakai: mojo run src/khq/calib.mojo <dir_dump> <out.bin>")
        return
    var path = String(args[1]) + "/kv_dump.bin"
    var out_path = String(args[2])
    print("== KHQ calib: baca", path)

    var nbytes_p = alloc[Int](1)
    nbytes_p[0] = 0
    var buf = slurp(path, nbytes_p)
    var nbytes = nbytes_p[0]
    nbytes_p.free()
    if nbytes < 16:
        print("!! dump kosong / tidak valid (", nbytes, " byte )")
        return
    if unpack_u32(buf, 0) != MAGIC_KHQD:
        print("!! magic dump salah")
        return
    var n_layers = Int(unpack_u32(buf, 4))
    var dim = Int(unpack_u32(buf, 8))
    print("   layers=", n_layers, " dim=", dim)

    # guard dump: tanpa ini, header korup -> alloc raksasa (malloc abort).
    # dim = num_key_value_heads * head_dim (lebar KV per token), jadi harus
    # kelipatan head_dim D — bukan sama dengan D (model nyata: 4*256 = 1024).
    if dim < D or dim % D != 0 or n_layers < 1 or n_layers > 256:
        print("!! dump tidak didukung: dim=", dim, " layers=", n_layers)
        return
    if 12 + 8 * n_layers > nbytes:
        print("!! dump terpotong: tabel header", 12 + 8 * n_layers,
              ">", nbytes, "byte")
        return

    var heads = dim // D
    var off = 12 + 8 * n_layers

    var out = open(out_path, "w")
    var hdr = alloc[UInt8](16)
    pack_u32(hdr, 0, MAGIC_KHQK)
    pack_u32(hdr, 4, 2)
    pack_u32(hdr, 8, UInt32(n_layers))
    pack_u32(hdr, 12, UInt32(D))
    _ = out.write_bytes(Span[UInt8, MutAnyOrigin](ptr=hdr, length=16))

    for li in range(n_layers):
        var layer_id = Int(unpack_u32(buf, 12 + li * 8))
        var n = Int(unpack_u32(buf, 12 + li * 8 + 4))
        if layer_id < 0 or layer_id >= 256 or n < 1 or n > 1 << 22:
            print("!! dump korup di layer indeks", li, ": layer_id=", layer_id,
                  " n=", n)
            out.close()
            hdr.free()
            return
        if off + 2 * n * dim * 4 > nbytes:
            print("!! dump terpotong di layer", layer_id, ": butuh",
                  off + 2 * n * dim * 4, "byte, ada", nbytes)
            out.close()
            hdr.free()
            return
        var rows = n * heads
        print("-- layer", layer_id, " token=", n, " sampel=", rows)
        var kptr = (buf + off).bitcast[Float32]()
        var vptr = (buf + off + n * dim * 4).bitcast[Float32]()
        off += n * dim * 4        # lewati blok K
        off += n * dim * 4        # lewati blok V

        # ---- susun (rows, D): heads dilipat jadi baris (head-major spt referensi)
        var k32 = alloc[Float32](rows * D)
        var k16 = alloc[Float32](rows * D)
        var v32 = alloc[Float32](rows * D)
        var v16 = alloc[Float32](rows * D)
        for t in range(n):
            for h in range(heads):
                for i in range(D):
                    var r = (t * heads + h) * D + i
                    k32[r] = kptr[t * dim + h * D + i]
                    k16[r] = fp16(kptr[t * dim + h * D + i])
                    v32[r] = vptr[t * dim + h * D + i]
                    v16[r] = fp16(vptr[t * dim + h * D + i])

        var subm = rows
        if subm > TOURN_ROWS:
            subm = TOURN_ROWS
        var subk32 = alloc[Float32](subm * D)
        var subk16 = alloc[Float32](subm * D)
        var subv32 = alloc[Float32](subm * D)
        var subv16 = alloc[Float32](subm * D)
        for i in range(subm * D):
            subk32[i] = k32[i]
            subk16[i] = k16[i]
            subv32[i] = v32[i]
            subv16[i] = v16[i]

        var gctx = DeviceContextGPU()
        var nul = UnsafePointer[Float32, MutAnyOrigin]()

        # ================= JALUR K (GPU) =================
        var dk = alloc[Float32](D)
        make_d(dk, layer_id)
        var rpk = alloc[Float32](NP * 16)
        for p in range(NP):
            for kk in range(4):
                for c in range(4):
                    rpk[p * 16 + kk * 4 + c] = Float32(1.0) if kk == c else Float32(0.0)
        var b_k32 = dev_f32(gctx, k32, rows * D)
        var b_dk = dev_f32(gctx, dk, D)
        var b_rpk = dev_f32(gctx, rpk, NP * 16)
        var b_normk = dev_new(gctx, rows)
        var b_projk = dev_new(gctx, rows * D)
        # proj = fwht(k32*d/norm)*f_s (rotor identity) — op 1
        if not cal_op(1, b_k32.unsafe_ptr(), b_dk.unsafe_ptr(), nul, nul,
                      b_normk.unsafe_ptr(), b_projk.unsafe_ptr(), nul,
                      0.0, 0.0, rows, 0, 0, 0):
            raise Error("kalibrasi GPU gagal: libbonsai_qmv_sm75.so tidak ada")
        var b_centsk = dev_new(gctx, 4 * D)
        var b_tsk = dev_new(gctx, 1)
        if not cal_op(2, b_projk.unsafe_ptr(), nul, nul, nul, nul,
                      b_centsk.unsafe_ptr(), b_tsk.unsafe_ptr(),
                      0.0, 0.0, rows, 0, 0, 0):
            raise Error("kalibrasi GPU gagal: libbonsai_qmv_sm75.so tidak ada")
        var one_h = alloc[Float32](2)
        d2h_f32(gctx, b_tsk, one_h, 1)
        var tsk = one_h[0]
        print("   K statis: ts=", tsk)
        # SmartK: grid 8x8 x 4 fase roundtrip compress/decompress, full GPU — op 8
        var b_subk32 = dev_f32(gctx, subk32, subm * D)
        var b_subk16 = dev_f32(gctx, subk16, subm * D)
        var b_centsk2 = dev_new(gctx, 4 * D)
        var b_tsak = dev_new(gctx, 2)
        if not cal_op(8, b_subk32.unsafe_ptr(), b_subk16.unsafe_ptr(),
                      b_dk.unsafe_ptr(), b_rpk.unsafe_ptr(),
                      b_centsk.unsafe_ptr(), b_centsk2.unsafe_ptr(),
                      b_tsak.unsafe_ptr(), tsk, 0.0, subm, 0, 0, 0):
            raise Error("SmartK GPU gagal: libbonsai_qmv_sm75.so tidak ada")
        d2h_f32(gctx, b_tsak, one_h, 2)
        var tsk2 = one_h[0]
        var alphak = one_h[1]
        var centsk2 = alloc[Float32](4 * D)
        d2h_f32(gctx, b_centsk2, centsk2, 4 * D)

        # ================= JALUR V (GPU) =================
        var dv = alloc[Float32](D)
        make_d_v(dv, layer_id)
        var b_dv = dev_f32(gctx, dv, D)
        var b_v32 = dev_f32(gctx, v32, rows * D)
        var b_vnorm = dev_new(gctx, rows)
        var b_vbase = dev_new(gctx, rows * D)
        # referensi memakai aktivasi fp32 utk kalibrasi V — op 1
        if not cal_op(1, b_v32.unsafe_ptr(), b_dv.unsafe_ptr(), nul, nul,
                      b_vnorm.unsafe_ptr(), b_vbase.unsafe_ptr(), nul,
                      0.0, 0.0, rows, 0, 0, 0):
            raise Error("kalibrasi GPU gagal: libbonsai_qmv_sm75.so tidak ada")
        var b_rpv = dev_new(gctx, NP * 16)
        # PCA per patch — op 4
        if not cal_op(4, b_vbase.unsafe_ptr(), nul, nul, nul,
                      b_rpv.unsafe_ptr(), nul, nul, 0.0, 0.0, rows, 0, 0, 0):
            raise Error("kalibrasi GPU gagal: libbonsai_qmv_sm75.so tidak ada")
        var b_centsv = dev_new(gctx, 4 * D)
        # cents awal dari data penuh — op 5
        cal_op(5, b_vbase.unsafe_ptr(), nul, nul, nul,
               b_centsv.unsafe_ptr(), nul, nul, 0.0, 0.0, rows, 0, 0, 0)
        var b_rot1 = dev_new(gctx, rows * D)
        # rotasi PCA (f32, tanpa pembulatan) — op 3
        cal_op(3, b_vbase.unsafe_ptr(), b_rpv.unsafe_ptr(), nul, nul,
               b_rot1.unsafe_ptr(), nul, nul, 0.0, 0.0, rows, 0, 0, 0)
        var b_vq = dev_new(gctx, NP * 768)
        # VQ 256x3 awal (seed layer_id*7919+13) — op 6
        cal_op(6, b_rot1.unsafe_ptr(), nul, nul, nul,
               b_vq.unsafe_ptr(), nul, nul, 0.0, 0.0, rows,
               layer_id * 7919 + 13, 0, 0)
        # 3 iterasi Procrustes-Lloyd (rp in/out device) — op 7
        cal_op(7, b_vbase.unsafe_ptr(), b_vq.unsafe_ptr(), nul, nul,
               b_rpv.unsafe_ptr(), nul, nul, 0.0, 0.0, rows, 3, 0, 0)
        var b_rotf = dev_new(gctx, rows * D)
        # cents & VQ final dari data ter-rotasi final
        cal_op(3, b_vbase.unsafe_ptr(), b_rpv.unsafe_ptr(), nul, nul,
               b_rotf.unsafe_ptr(), nul, nul, 0.0, 0.0, rows, 0, 0, 0)
        cal_op(5, b_rotf.unsafe_ptr(), nul, nul, nul,
               b_centsv.unsafe_ptr(), nul, nul, 0.0, 0.0, rows, 0, 0, 0)
        cal_op(6, b_rotf.unsafe_ptr(), nul, nul, nul,
               b_vq.unsafe_ptr(), nul, nul, 0.0, 0.0, rows,
               layer_id * 7919 + 13, 0, 0)

        # ---- SMARTVQ (train_smartvq.py) full GPU — v head-major device — op 10
        var b_vhm = dev_new(gctx, rows * D)
        cal_op(10, b_vbase.unsafe_ptr(), nul, nul, nul,
               b_vhm.unsafe_ptr(), nul, nul, 0.0, 0.0, n, heads, 0, 0)
        var hq_p = alloc[Int](1)
        var tp_p = alloc[Int](1)
        var attn = load_attn_layer(String(args[1]), layer_id, hq_p, tp_p)
        var use_attn = False
        if attn:
            if tp_p[0] != n:
                # Fallback senyap ke parseval = kalibrasi V tidak lagi
                # attention-aware. Lebih baik gagal jelas daripada hasil beda
                # diam-diam.
                raise Error(
                    "attn_" + String(layer_id) + ".bin punya " + String(tp_p[0])
                    + " token, dump punya " + String(n)
                    + " — kalibrasi dibatalkan (jangan fallback ke parseval)")
            use_attn = True
        if use_attn:
            print("   [SmartVQ] GPU mode attention-aware (H_q=", hq_p[0], ")")
            if not svq_gpu_run(
                gctx, b_vhm.unsafe_ptr(), rows, n, hq_p[0], heads,
                attn, b_vq.unsafe_ptr(), b_rpv.unsafe_ptr()):
                raise Error("SmartVQ GPU gagal: libbonsai_qmv_sm75.so tidak ada")
            attn.free()
        else:
            if not svq_gpu_run(
                gctx, b_vhm.unsafe_ptr(), rows, n, heads, heads,
                UnsafePointer[Float32, MutAnyOrigin](),
                b_vq.unsafe_ptr(), b_rpv.unsafe_ptr()):
                raise Error("SmartVQ GPU gagal: libbonsai_qmv_sm75.so tidak ada")
        hq_p.free()
        tp_p.free()

        # SmartV: centroid tetap, hanya alpha & ts — op 9
        var b_subv32 = dev_f32(gctx, subv32, subm * D)
        var b_subv16 = dev_f32(gctx, subv16, subm * D)
        var b_tsav = dev_new(gctx, 2)
        if not cal_op(9, b_subv32.unsafe_ptr(), b_subv16.unsafe_ptr(),
                      b_dv.unsafe_ptr(), b_rpv.unsafe_ptr(),
                      b_centsv.unsafe_ptr(), b_vq.unsafe_ptr(),
                      b_tsav.unsafe_ptr(), 1.95, 0.0, subm, 0, 0, 0):
            raise Error("SmartV GPU gagal: libbonsai_qmv_sm75.so tidak ada")
        d2h_f32(gctx, b_tsav, one_h, 2)
        var tsv = one_h[0]
        var alphav = one_h[1]

        # ================= TULIS BLOK LAYER =================
        var centsv_h = alloc[Float32](4 * D)
        var rpv_h = alloc[Float32](NP * 16)
        var vq_h = alloc[Float32](NP * 768)
        d2h_f32(gctx, b_centsv, centsv_h, 4 * D)
        d2h_f32(gctx, b_rpv, rpv_h, NP * 16)
        d2h_f32(gctx, b_vq, vq_h, NP * 768)
        var lb = alloc[UInt8](4)
        pack_u32(lb, 0, UInt32(layer_id))
        _ = out.write_bytes(Span[UInt8, MutAnyOrigin](ptr=lb.bitcast[UInt8](), length=4))
        var one = alloc[Float32](1)
        one[0] = tsk2
        write_f32(out, one, 1)
        one[0] = alphak
        write_f32(out, one, 1)
        write_f32(out, dk, D)
        write_f32(out, centsk2, 4 * D)
        write_f32(out, rpk, NP * 16)
        one[0] = tsv
        write_f32(out, one, 1)
        one[0] = alphav
        write_f32(out, one, 1)
        write_f32(out, dv, D)
        write_f32(out, centsv_h, 4 * D)
        write_f32(out, rpv_h, NP * 16)
        write_f32(out, vq_h, NP * 768)
        print("   V: alpha=", alphav, " ts=", tsv)

        k32.free()
        k16.free()
        v32.free()
        v16.free()
        subk32.free()
        subk16.free()
        subv32.free()
        subv16.free()
        dk.free()
        rpk.free()
        centsk2.free()
        dv.free()
        centsv_h.free()
        rpv_h.free()
        vq_h.free()
        one_h.free()
        lb.free()
        one.free()

    out.close()
    hdr.free()
    print("== selesai ->", out_path)

