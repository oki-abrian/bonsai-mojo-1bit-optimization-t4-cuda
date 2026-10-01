# ===----------------------------------------------------------------------=== #
# Module: src/models/qwen3_5/act_dump.mojo
# Purpose: Dump aktivasi kalibrasi (GSQ + S-CPFE/EoRA) dari jalur decode
#          biner Mojo — tiga lokasi per layer, meniru hook konverter MLX.
# ===----------------------------------------------------------------------=== #
# Opt-in via env BONSAI_DUMP_ACT_DIR (dikonfigurasi di main.mojo). Data hanya
# diambil dari DECODE (forward_gpu), bukan prefill — paritas dengan hook
# konverter yang aktif saat generate() (sampled generation), bukan prefill.
#
# Pemetaan site <-> hook konverter (convert_mxfp4_optimized.py):
#   site 0 "mlp_in"  <-> layer.mlp._cache_list                          (lebar D)
#   site 1 "attn_in" <-> layer.input_layernorm._attn_cache_list         (lebar D)
#   site 2 "ssm_in"  <-> layer.linear_attn.out_proj.
#                        _ssm_out_proj_input_cache               (lebar H_v*D_v)
#
# File keluaran per site <dir>/act_<nama>.bin (little-endian):
#   u32 magic 'ACTD' | u32 site | u32 n_layers | u32 width
#   tabel blok per layer: u32 layer_id | u32 n_tokens   (berurutan)
#   data   : per layer berurutan -> n_tokens * width nilai fp16
# Baris = token decode berurutan.
#
# Catatan volume: (D + D + H_v*D_v) * 2 byte/token/layer ~= 1,4 MB/token utk
# 64 layer (5120+5120+6144 elemen). JANGAN arahkan <dir> ke /kaggle/working
# bila data tidak ingin ikut sebagai output kernel (pelajaran KHQ: apa pun
# di situ menggemukkan arsip dan tidak kembali sebagai input).
#
# CATATAN SINTAKS (pelajaran build CPU 2026-09-26): `Scalar[dtype, origin]`
# TIDAK sah — Scalar (alias SIMD[dtype,1]) hanya menerima SATU parameter;
# origin hanya menjadi parameter UnsafePointer. Di dalam UnsafePointer
# tingkat mana pun tulis Scalar[DType.float16] atau alias Float16/Float32/
# UInt8 (bukti: khq_dump.mojo:47/162), dan bitcast hanya menerima SATU
# parameter, yaitu tipe pointer baru (bukti: khq_dump.mojo:182).

from memory import UnsafePointer, alloc
from gpu.host import DeviceBuffer
from .linear import DeviceContextGPU
from src.ops import copy_vec_sm75_launch_on, khq_state_slot_cell
from io.file import open

alias ACT_MAX_LAYERS = 64
alias ACT_N_SITES = 3


fn _act_site_name(site: Int) -> String:
    if site == 0:
        return "act_mlp_in.bin"
    if site == 1:
        return "act_attn_in.bin"
    return "act_ssm_in.bin"


struct ActDumpGlobals:
    var enabled: Bool
    var ready: Bool
    var dpath: String
    var cap: Int
    # per (site*ACT_MAX_LAYERS + layer): jumlah token yang sudah direkam
    var n: UnsafePointer[Int, MutAnyOrigin]
    # per site: lebar baris (ditetapkan pada rekaman pertama)
    var w: UnsafePointer[Int, MutAnyOrigin]
    # per (site*ACT_MAX_LAYERS + layer): buffer host fp16 (cap * width)
    var buf: UnsafePointer[UnsafePointer[Float16, MutAnyOrigin], MutAnyOrigin]
    var stage: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var host: UnsafePointer[Scalar[DType.float16], MutAnyOrigin]
    var stage_max: Int

    fn __init__(out self):
        self.enabled = False
        self.ready = False
        self.dpath = String()
        self.cap = 0
        self.n = UnsafePointer[Int, MutAnyOrigin]()
        self.w = UnsafePointer[Int, MutAnyOrigin]()
        self.buf = UnsafePointer[UnsafePointer[Float16, MutAnyOrigin], MutAnyOrigin]()
        self.stage = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.host = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.stage_max = 0


fn _ad() -> UnsafePointer[ActDumpGlobals, MutAnyOrigin]:
    """State modul via slot statik lib CUDA (slot 2; 0=KHQ runtime, 1=KHQ dump).
    Bentuk identik dgn _dg() khq_dump: sel dimakna langsung sebagai pointer ke
    pointer ke struct. Dipanggil hanya dari jalur dump (env diset), jadi lib
    CUDA dipastikan ada — tanpa penjaga null, seperti khq.
    Hasil FFI ditampung lewat alamat variabel lokal — UnsafePointer(to=...),
    pengganti resmi address_of yang dihapus sejak Mojo 25.5 — lihat komentar
    khq_state_slot_cell di src/ops.mojo (jebakan tail call + dlclose,
    build CPU)."""
    var tmp = UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin]()
    var sel = UnsafePointer[UnsafePointer[UnsafePointer[UInt8, MutAnyOrigin], MutAnyOrigin], MutAnyOrigin](to=tmp)
    khq_state_slot_cell(sel, 2)
    var cell = tmp.bitcast[UnsafePointer[ActDumpGlobals, MutAnyOrigin]]()
    var p = cell[]
    if p == UnsafePointer[ActDumpGlobals, MutAnyOrigin]():
        p = alloc[ActDumpGlobals](1)
        p[] = ActDumpGlobals()
        cell[] = p
    return p


fn _pack_u32_act(p: UnsafePointer[UInt8, MutAnyOrigin], v: UInt32):
    p[0] = UInt8(v & 0xFF)
    p[1] = UInt8((v >> 8) & 0xFF)
    p[2] = UInt8((v >> 16) & 0xFF)
    p[3] = UInt8((v >> 24) & 0xFF)


fn act_dump_configure(dir_path: String, max_tokens: Int) raises:
    var p = _ad()
    p[].enabled = True
    p[].dpath = dir_path
    p[].cap = max_tokens
    p[].n = alloc[Int](ACT_N_SITES * ACT_MAX_LAYERS)
    p[].w = alloc[Int](ACT_N_SITES)
    p[].buf = alloc[UnsafePointer[Float16, MutAnyOrigin]](ACT_N_SITES * ACT_MAX_LAYERS)
    for i in range(ACT_N_SITES * ACT_MAX_LAYERS):
        p[].n[i] = 0
        p[].buf[i] = UnsafePointer[Float16, MutAnyOrigin]()
    for s in range(ACT_N_SITES):
        p[].w[s] = 0


fn act_dump_site(
    mut ctx: DeviceContextGPU,
    layer_idx: Int,
    site: Int,
    src_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    width: Int,
) raises:
    """Rekam satu baris aktivasi (satu token) dari VRAM ke buffer host.
    Dipanggil dari forward_gpu di titik yang nilainya masih sah (sebelum
    buffer sumber ditimpa tahap berikutnya)."""
    var p = _ad()
    if not p[].enabled:
        return
    if layer_idx < 0 or layer_idx >= ACT_MAX_LAYERS:
        return
    if site < 0 or site >= ACT_N_SITES:
        return
    var idx = site * ACT_MAX_LAYERS + layer_idx
    if p[].n[idx] >= p[].cap:
        return

    # Lebar site dikunci pada rekaman pertama (semua layer satu site sama).
    if p[].w[site] == 0:
        p[].w[site] = width
    elif p[].w[site] != width:
        raise Error(
            "ACT-DUMP: lebar site " + String(site) + " berubah: "
            + String(p[].w[site]) + " vs " + String(width)
        )

    # Staging device + host: dialokasi sekali, diganti bila lebar membesar
    # (buffer lama dibiarkan — ukurannya puluhan KB, mode kalibrasi saja).
    if not p[].ready or width > p[].stage_max:
        p[].stage = alloc[DeviceBuffer[DType.float16]](1)
        p[].stage.init_pointee_move(ctx.enqueue_create_buffer[DType.float16](width))
        p[].host = alloc[Scalar[DType.float16]](width)
        p[].stage_max = width
        p[].ready = True

    copy_vec_sm75_launch_on[DType.float16](
        ctx, p[].stage[].unsafe_ptr(), src_dev, width
    )
    ctx.synchronize()
    ctx.enqueue_copy(p[].host, p[].stage[])
    ctx.synchronize()

    if p[].buf[idx] == UnsafePointer[Float16, MutAnyOrigin]():
        p[].buf[idx] = alloc[Float16](p[].cap * width)
    var n_tok = p[].n[idx]
    var dst = p[].buf[idx] + n_tok * width
    for i in range(width):
        dst[i] = Float16(p[].host[i])
    p[].n[idx] = n_tok + 1


fn act_dump_flush() raises:
    """Tulis seluruh site yang berisi data ke <dir>/act_<nama>.bin lalu
    bebaskan buffer host."""
    var p = _ad()
    if not p[].enabled:
        return
    p[].enabled = False
    for site in range(ACT_N_SITES):
        var n_layers = 0
        for li in range(ACT_MAX_LAYERS):
            if p[].n[site * ACT_MAX_LAYERS + li] > 0:
                n_layers += 1
        if n_layers == 0:
            continue
        var width = p[].w[site]

        var path = p[].dpath + "/" + _act_site_name(site)
        var f = open(path, "w")
        var hdr = alloc[UInt8](16 + 8 * n_layers)
        _pack_u32_act(hdr, 0x44544341)  # 'ACTD'
        _pack_u32_act(hdr + 4, UInt32(site))
        _pack_u32_act(hdr + 8, UInt32(n_layers))
        _pack_u32_act(hdr + 12, UInt32(width))
        var off = 16
        var n_tok_total = 0
        for li in range(ACT_MAX_LAYERS):
            var n_tok = p[].n[site * ACT_MAX_LAYERS + li]
            if n_tok == 0:
                continue
            _pack_u32_act(hdr + off, UInt32(li))
            _pack_u32_act(hdr + off + 4, UInt32(n_tok))
            off += 8
            n_tok_total += n_tok
        _ = f.write_bytes(Span[UInt8, MutAnyOrigin](ptr=hdr, length=16 + 8 * n_layers))
        hdr.free()

        for li in range(ACT_MAX_LAYERS):
            var idx = site * ACT_MAX_LAYERS + li
            var n_tok = p[].n[idx]
            if n_tok == 0:
                continue
            var nb = n_tok * width * 2
            _ = f.write_bytes(Span[UInt8, MutAnyOrigin](
                ptr=p[].buf[idx].bitcast[UInt8](), length=nb))
            p[].buf[idx].free()
            p[].buf[idx] = UnsafePointer[Float16, MutAnyOrigin]()
            p[].n[idx] = 0
        f.close()
        print(">> [ACT-DUMP] site", site, _act_site_name(site),
              "->", path, "|", n_layers, "layer |", n_tok_total, "token |",
              "lebar", width)
