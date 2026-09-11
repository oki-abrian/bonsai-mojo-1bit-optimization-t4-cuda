# ===----------------------------------------------------------------------=== #
# Script: main.mojo
# Purpose: CLI INFERENSI NATIVE MOJO (tanpa Python) untuk Bonsai-27B-mlx-1bit:
#          muat config.json + safetensors via loader Mojo, bangun 64 layer
#          hybrid Qwen3.5 (3 GDN : 1 Gated Attention), generasi greedy.
#          Prompt diberikan sebagai token id (v1); tokenisasi BPE menyusul.
# Catatan: ini harness korektess FP32 host-sim — decode 27B di CPU lambat
#          (~detik per token); performa datang dari jalur GPU MAX.
# Risiko kompilasi terflag: String(StringSpan), Span[UInt8, MutAnyOrigin](ptr=ptr, length=n),
#          FileHandle.seek(offset, whence), `from io.file import open`,
#          assignment elemen UnsafePointer[Struct, MutAnyOrigin] (lihat NOTES_API_*.md).
# ===----------------------------------------------------------------------=== #

from sys import argv
from os import setenv, getenv
from memory import UnsafePointer, alloc
from src import (
    QwenConfig, QwenDecoderLayer, QwenLinear1Bit, GatedDeltaNetState,
    AttentionKVCache, qwen3_5_model_forward,
    khq_dump_configure, khq_dump_flush,
    khq_active, khq_activate
)
from time import monotonic
from src.jsonlite import JsonDoc
from src.safetensors import SafeTensorsIndex, fuse_u8, fuse_f32, load_qlinear
from src.models.qwen3_5.linear import use_gpu_matmul, DeviceContextGPU
from src.models.qwen3_5.gpu_ctx import gpu_ctx_new
from gpu.host import DeviceBuffer
from src.ops import (
    rmsnorm_sm75_launch_on, argmax_sm75_launch_on,
    embed_lookup_1bit_sm75_launch_on, copy_vec_sm75_launch_on
)

fn dump_top2_prefill(
    gpu_ctx_ptr: UnsafePointer[DeviceContextGPU, MutAnyOrigin],
    logits_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    V: Int
) raises:
    """Diagnostik near-tie: top-2 logit + gap di batas prefill. Gap kecil
    (<~0.05) + flip token vs jalur per-token = wajar (beda urutan akumulasi
    fp16 antara WMMA vs GEMV); gap besar = indikasi bug layout."""
    var ctx = gpu_ctx_ptr[]
    var tmp = ctx.enqueue_create_buffer[DType.float16](V)
    copy_vec_sm75_launch_on[DType.float16](
        ctx, tmp.unsafe_ptr(), logits_dev, V
    )
    ctx.synchronize()
    var hlog = alloc[Scalar[DType.float16]](V)
    ctx.enqueue_copy(hlog, tmp)
    ctx.synchronize()
    var b1 = 0
    var b2 = 0
    var v1: Float64 = -1e30
    var v2: Float64 = -1e30
    for i in range(V):
        var lv = hlog[i].cast[DType.float64]()
        if lv > v1:
            v2 = v1
            b2 = b1
            v1 = lv
            b1 = i
        elif lv > v2:
            v2 = lv
            b2 = i
    print(">> [TOP2] 1st:", b1, "=", v1, "| 2nd:", b2, "=", v2,
          "| gap:", v1 - v2)
    hlog.free()


fn find_flex(index: SafeTensorsIndex, name: String) -> Int:
    """Cari tensor: nama apa adanya, lalu fallback prefix 'language_model.'
    (repo Bonsai-27B-mlx-1bit dikemas ala VLM Qwen3.5)."""
    var e = index.find(name)
    if e != -1:
        return e
    return index.find("language_model." + name)

fn read_scales_eff(index: SafeTensorsIndex, e_s: Int, numel: Int) raises -> UnsafePointer[Float32, MutAnyOrigin]:
    """Scales efektif = s_checkpoint / 2 (untuk kernel (2q-1)*s_eff)."""
    var p = alloc[Float32](numel if numel > 0 else 1)
    if e_s != -1 and numel > 0:
        var _ = index.read_f32(e_s, p, numel)
        for i in range(numel):
            p[i] *= Float32(0.5)
    return p

fn read_biases_f32(index: SafeTensorsIndex, e_b: Int, numel: Int) raises -> UnsafePointer[Float32, MutAnyOrigin]:
    """Biases checkpoint mentah (w = q*s_ckpt + b, affine eksak)."""
    var p = alloc[Float32](numel if numel > 0 else 1)
    if e_b != -1 and numel > 0:
        var _ = index.read_f32(e_b, p, numel)
    return p

fn embed_row_dequant(
    hidden: UnsafePointer[Float32, MutAnyOrigin],
    packed: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    biases: UnsafePointer[Float32, MutAnyOrigin],
    token: Int,
    D: Int
):
    """Dequant SATU baris embedding 1-bit affine (w = q*s + b) langsung ke
    hidden — paritas mx.dequantize(weight[ids], scales[ids], biases[ids])."""
    var base = token * (D // 8)
    var srow = token * (D // 128)
    for k in range(D):
        var bit = Float32((Int(packed[base + (k >> 3)]) >> (k & 7)) & 1)
        hidden[k] = bit * scales[srow + (k >> 7)] + biases[srow + (k >> 7)]

fn lm_head_argmax_packed(
    hidden: UnsafePointer[Float32, MutAnyOrigin],
    packed: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    biases: UnsafePointer[Float32, MutAnyOrigin],
    V: Int,
    D: Int
) -> Int:
    """Argmax logits = lm_head @ hidden langsung dari data 1-bit terpaket —
    dequant affine (q*s + b) di dalam loop komputasi (kernel non-fused)."""
    var best = Float32(-3.0e38)
    var best_v = 0
    for v in range(V):
        var rowb = v * (D // 8)
        var srow = v * (D // 128)
        var acc: Float32 = 0.0
        for k in range(D):
            var bit = Float32((Int(packed[rowb + (k >> 3)]) >> (k & 7)) & 1)
            acc += (bit * scales[srow + (k >> 7)] + biases[srow + (k >> 7)]) * hidden[k]
        if acc > best:
            best = acc
            best_v = v
    return best_v

fn argmax_f32(
    x: UnsafePointer[Float32, MutAnyOrigin],
    n: Int
) -> Int:
    var best = x[0]
    var bi = 0
    for i in range(1, n):
        if x[i] > best:
            best = x[i]
            bi = i
    return bi


fn parse_int_list(spec: String) -> UnsafePointer[Int, MutAnyOrigin]:
    """Parse "1,2,3" -> array Int; slot 0 menyimpan jumlah elemen,
    elemen token mulai slot 1."""
    var bytes = spec.as_bytes()
    var n = len(bytes)
    var count = 1
    for i in range(n):
        if bytes[i] == UInt8(ord(",")):
            count += 1
    var out = alloc[Int](count + 1)
    var v = 0
    var idx = 1 # slot 0 dicadangkan untuk jumlah
    var has_digits = False
    for i in range(n):
        var c = bytes[i]
        if c == UInt8(ord(",")):
            out[idx] = v
            idx += 1
            v = 0
            has_digits = False
        elif c >= UInt8(ord("0")) and c <= UInt8(ord("9")):
            v = v * 10 + Int(c) - Int(ord("0"))
            has_digits = True
    if has_digits:
        out[idx] = v
        idx += 1
    out[0] = idx - 1 # jumlah elemen efektif
    return out

fn read_small_file(path: String, out_buf: UnsafePointer[UInt8, MutAnyOrigin], cap: Int) raises -> Int:
    """Baca file teks kecil (config.json) — kembalikan jumlah byte terbaca."""
    var f = open(path, "r")
    var n = f.read(Span[UInt8, MutAnyOrigin](ptr=out_buf, length=cap))
    f.close()
    return n

fn main() raises:
    # ---------------- 0. Argumen CLI ----------------
    var args = argv()
    var model_dir = String()
    var prompt_spec = String()
    var max_tokens = 64
    var have_model_dir = False
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--model-dir" and i + 1 < len(args):
            i += 1
            model_dir = String(args[i])
            have_model_dir = True
        elif a == "--prompt-tokens" and i + 1 < len(args):
            i += 1
            prompt_spec = String(args[i])
        elif a == "--max-tokens" and i + 1 < len(args):
            i += 1
            max_tokens = Int(args[i])
        elif a == "--gpu":
            _ = setenv("BONSAI_USE_GPU", "1", 1)
        i += 1

    if not have_model_dir or len(prompt_spec) == 0:
        print("Pemakaian: mojo run main.mojo -- --model-dir <dir> --prompt-tokens <id,id,...> --max-tokens <n> [--gpu]")
        return

    if use_gpu_matmul():
        print(">> [MOJO-NATIVE] Inferensi Bonsai-27B-mlx-1bit (JALUR GPU T4)")
    else:
        print(">> [MOJO-NATIVE] Inferensi Bonsai-27B-mlx-1bit (FP32 host-sim)")
        print(">> CATATAN: decode 27B di CPU lambat (validasi kebenaran); performa via jalur GPU MAX")

    # ---------------- 1. Config ----------------
    var cfg = QwenConfig.qwen_27b_default()
    var cfg_path = model_dir + "/" + "config.json"
    var cfg_buf = alloc[UInt8](1 << 20)
    var cfg_len = read_small_file(cfg_path, cfg_buf, 1 << 20)
    if cfg_len > 0:
        var cdoc = JsonDoc()
        if cdoc.parse_bytes(cfg_buf, cfg_len):
            var r = cdoc.root()
            # Repo asli = kemasan VLM: parameter teks bersarang di 'text_config'
            var tcfg = cdoc.obj_get(r, "text_config")
            if tcfg != -1:
                r = tcfg
            var nd = cdoc.obj_get(r, "hidden_size")
            if nd != -1:
                cfg.hidden_size = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "intermediate_size")
            if nd != -1:
                cfg.intermediate_size = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "num_hidden_layers")
            if nd != -1:
                cfg.num_hidden_layers = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "num_attention_heads")
            if nd != -1:
                cfg.num_attention_heads = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "num_key_value_heads")
            if nd != -1:
                cfg.num_key_value_heads = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "head_dim")
            if nd != -1:
                cfg.head_dim = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "vocab_size")
            if nd != -1:
                cfg.vocab_size = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "rms_norm_eps")
            if nd != -1:
                cfg.rms_norm_eps = Float32(cdoc.as_f64(nd))
            nd = cdoc.obj_get(r, "rope_theta")
            if nd != -1:
                cfg.rope_theta = Float32(cdoc.as_f64(nd))
            nd = cdoc.obj_get(r, "partial_rotary_factor")
            if nd != -1:
                cfg.partial_rotary_factor = Float32(cdoc.as_f64(nd))
                cfg.rotary_dim = Int(Float32(cfg.head_dim) * cfg.partial_rotary_factor)
            # rope_theta ASLI ada di objek bersarang "rope_parameters"
            # (qwen3_5: 10000000) — key flat "rope_theta" TIDAK ADA di
            # text_config; tanpa ini RoPE memakai default 100000 (salah).
            var rp = cdoc.obj_get(r, "rope_parameters")
            if rp != -1:
                var rt = cdoc.obj_get(rp, "rope_theta")
                if rt != -1:
                    cfg.rope_theta = Float32(cdoc.as_f64(rt))
            nd = cdoc.obj_get(r, "linear_num_value_heads")
            if nd != -1:
                cfg.gdn_num_v_heads = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "linear_num_key_heads")
            if nd != -1:
                cfg.gdn_num_k_heads = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "linear_key_head_dim")
            if nd != -1:
                cfg.gdn_head_k_dim = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "linear_value_head_dim")
            if nd != -1:
                cfg.gdn_head_v_dim = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "linear_conv_kernel_dim")
            if nd != -1:
                cfg.gdn_conv_kernel = cdoc.as_int(nd)
            cfg.gdn_conv_dim = 2 * (cfg.gdn_num_k_heads * cfg.gdn_head_k_dim) + (cfg.gdn_num_v_heads * cfg.gdn_head_v_dim)
        cdoc.free()
    cfg_buf.free()

    var D = cfg.hidden_size
    var V = cfg.vocab_size
    print(">> Config: layers=", cfg.num_hidden_layers, " hidden=", D, " vocab=", V)

    # ---------------- 2. Buka safetensors ----------------
    print(">> [STEP] open_dir...")
    var index = SafeTensorsIndex()
    index.open_dir(model_dir)
    if not index.ok:
        print(">> [ERROR] Tidak ada tensor safetensors ditemukan di", model_dir)
        return
    print(">> [STEP] indeks:", index.n_entries, "tensor /", index.n_shards, "shard")

    # ---------------- 3. Muat bobot global (TETAP 1-BIT TERPAKET) ----------------
    # Mekanisme MLX/llama.cpp Q1_0_g128: bobot hidup di memori terpaket;
    # dequant (2q-1)*s terjadi DI DALAM komputasi (per baris embed, per
    # elemen di loop/kernel matmul) — TIDAK ada tabel FP32 raksasa.
    print(">> [STEP] cari embed...")
    var e_emb = find_flex(index, "model.embed_tokens.weight")
    var e_emb_s = find_flex(index, "model.embed_tokens.scales")
    if e_emb == -1 or e_emb_s == -1:
        print(">> [ERROR] embed_tokens/scales tidak ditemukan (kedua prefix)")
        return
    var V_emb = index.entries[e_emb].d0
    var D_real = index.entries[e_emb].d1 * 32 # kata U32 -> elemen (bits=1)
    var embed_w = alloc[UInt8](index.entries[e_emb].nbytes)
    if not index.read_raw(e_emb, embed_w, index.entries[e_emb].nbytes):
        print(">> [ERROR] baca embed gagal")
        return
    var embed_s = alloc[Float32](V_emb * (D_real // 128))
    if not index.read_f32(e_emb_s, embed_s, V_emb * (D_real // 128)):
        print(">> [ERROR] baca scales embed gagal")
        return
    var e_emb_b = find_flex(index, "model.embed_tokens.biases")
    var embed_b = alloc[Float32](V_emb * (D_real // 128))
    if e_emb_b != -1:
        var _ = index.read_f32(e_emb_b, embed_b, V_emb * (D_real // 128))
    D = D_real
    V = V_emb
    cfg.hidden_size = D
    cfg.vocab_size = V

    var final_norm_w = alloc[Float32](D)
    var e_fnorm = find_flex(index, "model.norm.weight")
    if e_fnorm != -1:
        var _ = index.read_f32(e_fnorm, final_norm_w, D)

    var e_lm = find_flex(index, "lm_head.weight")
    var e_lm_s = find_flex(index, "lm_head.scales")
    if e_lm == -1 or e_lm_s == -1:
        print(">> [ERROR] lm_head/scales tidak ditemukan")
        return
    var V_lm = index.entries[e_lm].d0
    var lm_w = alloc[UInt8](index.entries[e_lm].nbytes)
    if not index.read_raw(e_lm, lm_w, index.entries[e_lm].nbytes):
        print(">> [ERROR] baca lm_head gagal")
        return
    var lm_s = alloc[Float32](V_lm * (D // 128))
    if not index.read_f32(e_lm_s, lm_s, V_lm * (D // 128)):
        print(">> [ERROR] baca scales lm_head gagal")
        return
    var e_lm_b = find_flex(index, "lm_head.biases")
    var lm_b = alloc[Float32](V_lm * (D // 128))
    if e_lm_b != -1:
        var _ = index.read_f32(e_lm_b, lm_b, V_lm * (D // 128))
    # Kontrak kernel: scales_eff = s_ckpt / 2 (kernel menghitung (2q-1)*s_eff).
    for i in range(V_lm * (D // 128)):
        lm_s[i] = lm_s[i] * 0.5
    var lm_proj = QwenLinear1Bit(lm_w, lm_s, lm_b, V_lm, D)
    V = V_lm
    print(">> [STEP] bobot global siap (packed 1-bit, V=", V, " D=", D, ")")

    # ---------------- 4. Muat bobot per layer (dengan fusion Paket C) ----------------
    var n_layers = cfg.num_hidden_layers

    # KHQ: dump K/V untuk kalibrasi (env BONSAI_DUMP_KV_DIR) — K unroped.
    var khq_dir = getenv("BONSAI_DUMP_KV_DIR")
    if khq_dir:
        khq_dump_configure(
            khq_dir, 2048, cfg.num_key_value_heads * cfg.head_dim
        )
        print(">> [KHQ-DUMP] aktif ->", khq_dir)

    # KHQ: aktivasi dipindah ke setelah DeviceContext terpasang (lihat bawah).

    var layers = alloc[QwenDecoderLayer](n_layers)
    var gdn_states = alloc[GatedDeltaNetState](n_layers)
    var kv_caches = alloc[AttentionKVCache](n_layers)
    # pemetaan layer -> indeks state (GDN) / cache (attention)
    var gdn_idx = alloc[Int](n_layers)
    var kv_idx = alloc[Int](n_layers)
    var n_gdn = 0
    var n_kv = 0
    var max_seq = 4096

    for li in range(n_layers):
        var is_linear = (li % cfg.full_attention_interval) != (cfg.full_attention_interval - 1)
        layers[li] = QwenDecoderLayer(layer_idx=li, is_linear=is_linear, config=cfg)
        var prefix = "model.layers." + String(li) + "."
        # norm
        var ln1 = alloc[Float32](D)
        var e_ln1 = find_flex(index, prefix + "input_layernorm.weight")
        if e_ln1 != -1:
            var _ = index.read_f32(e_ln1, ln1, D)
        var ln2 = alloc[Float32](D)
        var e_ln2 = find_flex(index, prefix + "post_attention_layernorm.weight")
        if e_ln2 != -1:
            var _ = index.read_f32(e_ln2, ln2, D)
        layers[li].input_layernorm_w = ln1
        layers[li].post_attn_layernorm_w = ln2

        if is_linear:
            # GDN: in_proj_all (fused) atau fusion manual qkv+z+b+a
            var w_all = UnsafePointer[UInt8, MutAnyOrigin]()
            var s_all = UnsafePointer[Float32, MutAnyOrigin]()
            var b_all = UnsafePointer[Float32, MutAnyOrigin]()
            var n_all = 0
            var k_all = 0
            var e_fused_w = find_flex(index, prefix + "linear_attn.in_proj_all.weight")
            if e_fused_w != -1:
                var nb = index.entries[e_fused_w].nbytes
                w_all = alloc[UInt8](nb)
                var _ = index.read_raw(e_fused_w, w_all, nb)
                n_all = index.dim0(e_fused_w)
                k_all = nb * 8 // n_all
                var e_fused_s = find_flex(index, prefix + "linear_attn.in_proj_all.scales")
                s_all = read_scales_eff(index, e_fused_s, n_all * (k_all // 128))
                var e_fused_b = find_flex(index, prefix + "linear_attn.in_proj_all.biases")
                b_all = read_biases_f32(index, e_fused_b, n_all * (k_all // 128))
            else:
                # fusion manual 4 proyeksi (paritas loader.py)
                var pq = load_qlinear(index, prefix + "linear_attn.in_proj_qkv.weight", prefix + "linear_attn.in_proj_qkv.scales", prefix + "linear_attn.in_proj_qkv.biases")
                var pz = load_qlinear(index, prefix + "linear_attn.in_proj_z.weight", prefix + "linear_attn.in_proj_z.scales", prefix + "linear_attn.in_proj_z.biases")
                var pb = load_qlinear(index, prefix + "linear_attn.in_proj_b.weight", prefix + "linear_attn.in_proj_b.scales", prefix + "linear_attn.in_proj_b.biases")
                var pa = load_qlinear(index, prefix + "linear_attn.in_proj_a.weight", prefix + "linear_attn.in_proj_a.scales", prefix + "linear_attn.in_proj_a.biases")
                if not (pq.ok and pz.ok and pb.ok and pa.ok):
                    print(">> [ERROR] bobot GDN tidak lengkap pada layer", li)
                    return
                var w01 = fuse_u8(pq.w, pq.nbytes, pz.w, pz.nbytes)
                var w23 = fuse_u8(pb.w, pb.nbytes, pa.w, pa.nbytes)
                w_all = fuse_u8(w01, pq.nbytes + pz.nbytes, w23, pb.nbytes + pa.nbytes)
                var s01 = fuse_f32(pq.scales, pq.n_rows * (pq.k_dim // 128), pz.scales, pz.n_rows * (pz.k_dim // 128))
                var s23 = fuse_f32(pb.scales, pb.n_rows * (pb.k_dim // 128), pa.scales, pa.n_rows * (pa.k_dim // 128))
                s_all = fuse_f32(s01, (pq.n_rows + pz.n_rows) * (pq.k_dim // 128), s23, (pb.n_rows + pa.n_rows) * (pb.k_dim // 128))
                var b01 = fuse_f32(pq.biases, pq.n_rows * (pq.k_dim // 128), pz.biases, pz.n_rows * (pz.k_dim // 128))
                var b23 = fuse_f32(pb.biases, pb.n_rows * (pb.k_dim // 128), pa.biases, pa.n_rows * (pa.k_dim // 128))
                b_all = fuse_f32(b01, (pq.n_rows + pz.n_rows) * (pq.k_dim // 128), b23, (pb.n_rows + pa.n_rows) * (pb.k_dim // 128))
                n_all = pq.n_rows + pz.n_rows + pb.n_rows + pa.n_rows
                k_all = pq.k_dim
            layers[li].gdn_in_proj_all = QwenLinear1Bit(w_all, s_all, b_all, n_all, k_all)

            # conv1d weight -> FP32
            var cw = alloc[Float32](cfg.gdn_conv_dim * cfg.gdn_conv_kernel)
            var e_cw = find_flex(index, prefix + "linear_attn.conv1d.weight")
            if e_cw != -1:
                var _ = index.read_f32(e_cw, cw, cfg.gdn_conv_dim * cfg.gdn_conv_kernel)
            layers[li].gdn_conv_weights = cw

            # out_proj
            var e_ow = find_flex(index, prefix + "linear_attn.out_proj.weight")
            var ow = UnsafePointer[UInt8, MutAnyOrigin]()
            var os_ = UnsafePointer[Float32, MutAnyOrigin]()
            var ob_ = UnsafePointer[Float32, MutAnyOrigin]()
            var on = 0
            var ok_ = 0
            if e_ow != -1:
                var nb = index.entries[e_ow].nbytes
                ow = alloc[UInt8](nb)
                var _ = index.read_raw(e_ow, ow, nb)
                on = index.dim0(e_ow)
                ok_ = nb * 8 // on
                var e_os = find_flex(index, prefix + "linear_attn.out_proj.scales")
                os_ = read_scales_eff(index, e_os, on * (ok_ // 128))
                var e_ob = find_flex(index, prefix + "linear_attn.out_proj.biases")
                ob_ = read_biases_f32(index, e_ob, on * (ok_ // 128))
            layers[li].gdn_out_proj = QwenLinear1Bit(ow, os_, ob_, on, ok_)

            # Parameter riil Qwen3-Next: A_log, dt_bias, norm GDN
            var e_al = find_flex(index, prefix + "linear_attn.A_log")
            if e_al != -1:
                var nn_al = index.entries[e_al].d0 * max(index.entries[e_al].d1, 1)
                var a_log = alloc[Float32](nn_al)
                var _ = index.read_f32(e_al, a_log, nn_al)
                layers[li].gdn_a_log = a_log
            var e_dt = find_flex(index, prefix + "linear_attn.dt_bias")
            if e_dt != -1:
                var nn_dt = index.entries[e_dt].d0 * max(index.entries[e_dt].d1, 1)
                var dtb = alloc[Float32](nn_dt)
                var _ = index.read_f32(e_dt, dtb, nn_dt)
                layers[li].gdn_dt_bias = dtb
            var e_gn = find_flex(index, prefix + "linear_attn.norm.weight")
            if e_gn != -1:
                var nn_gn = index.entries[e_gn].d0 * max(index.entries[e_gn].d1, 1)
                var gnw = alloc[Float32](nn_gn)
                var _ = index.read_f32(e_gn, gnw, nn_gn)
                layers[li].gdn_norm_w = gnw
            if e_al != -1 and e_dt != -1 and e_gn != -1:
                layers[li].gdn_has_params = True

            # state GDN + pemetaan
            gdn_states[n_gdn] = GatedDeltaNetState(
                cfg.gdn_conv_dim, cfg.gdn_num_v_heads,
                cfg.gdn_head_v_dim, cfg.gdn_head_k_dim
            )
            gdn_idx[li] = n_gdn
            n_gdn += 1
            kv_idx[li] = -1
        else:
            # Gated Full Attention: q/k/v/o (eksplisit, paritas loader.py)
            var pq = load_qlinear(index, prefix + "self_attn.q_proj.weight", prefix + "self_attn.q_proj.scales", prefix + "self_attn.q_proj.biases")
            var pk = load_qlinear(index, prefix + "self_attn.k_proj.weight", prefix + "self_attn.k_proj.scales", prefix + "self_attn.k_proj.biases")
            var pv = load_qlinear(index, prefix + "self_attn.v_proj.weight", prefix + "self_attn.v_proj.scales", prefix + "self_attn.v_proj.biases")
            var po = load_qlinear(index, prefix + "self_attn.o_proj.weight", prefix + "self_attn.o_proj.scales", prefix + "self_attn.o_proj.biases")
            if not (pq.ok and pk.ok and pv.ok and po.ok):
                print(">> [ERROR] bobot attention tidak lengkap pada layer", li)
                return
            layers[li].attn_q_proj = QwenLinear1Bit(pq.w, pq.scales, pq.biases, pq.n_rows, pq.k_dim)
            layers[li].attn_k_proj = QwenLinear1Bit(pk.w, pk.scales, pk.biases, pk.n_rows, pk.k_dim)
            layers[li].attn_v_proj = QwenLinear1Bit(pv.w, pv.scales, pv.biases, pv.n_rows, pv.k_dim)
            layers[li].attn_o_proj = QwenLinear1Bit(po.w, po.scales, po.biases, po.n_rows, po.k_dim)

            # Bobot Q-Norm & K-Norm per head (paritas mlx-lm qwen3_next)
            var e_qn = find_flex(index, prefix + "self_attn.q_norm.weight")
            var nn_q = 0
            if e_qn != -1:
                nn_q = index.entries[e_qn].d0 * max(index.entries[e_qn].d1, 1)
                var qnw = alloc[Float32](nn_q)
                var _ = index.read_f32(e_qn, qnw, nn_q)
                layers[li].attn_q_norm_w = qnw
            var e_kn = find_flex(index, prefix + "self_attn.k_norm.weight")
            var nn_k = 0
            if e_kn != -1:
                nn_k = index.entries[e_kn].d0 * max(index.entries[e_kn].d1, 1)
                var knw = alloc[Float32](nn_k)
                var _ = index.read_f32(e_kn, knw, nn_k)
                layers[li].attn_k_norm_w = knw
            if e_qn != -1 and e_kn != -1:
                layers[li].attn_has_norms = True
            if li == 3:
                print(">> [ATTN LOAD] Layer", li, "attn_has_norms:", layers[li].attn_has_norms, "nn_q:", nn_q, "nn_k:", nn_k)
            # KV cache + pemetaan
            kv_caches[n_kv] = AttentionKVCache(
                max_seq, cfg.num_key_value_heads, cfg.head_dim
            )
            kv_idx[li] = n_kv
            n_kv += 1
            gdn_idx[li] = -1

        # MLP SwiGLU: gate_up fused atau gate+up
        var e_guw = find_flex(index, prefix + "mlp.gate_up_proj.weight")
        if e_guw != -1:
            var nb = index.entries[e_guw].nbytes
            var wb = alloc[UInt8](nb)
            var _ = index.read_raw(e_guw, wb, nb)
            var nn = index.dim0(e_guw)
            var kk = nb * 8 // nn
            var e_gus = find_flex(index, prefix + "mlp.gate_up_proj.scales")
            var sb = read_scales_eff(index, e_gus, nn * (kk // 128))
            var e_gub = find_flex(index, prefix + "mlp.gate_up_proj.biases")
            var gb = read_biases_f32(index, e_gub, nn * (kk // 128))
            layers[li].mlp_gate_up_proj = QwenLinear1Bit(wb, sb, gb, nn, kk)
        else:
            # MLP: fusion gate+up (paritas loader.py)
            var pg = load_qlinear(index, prefix + "mlp.gate_proj.weight", prefix + "mlp.gate_proj.scales", prefix + "mlp.gate_proj.biases")
            var pu = load_qlinear(index, prefix + "mlp.up_proj.weight", prefix + "mlp.up_proj.scales", prefix + "mlp.up_proj.biases")
            if not (pg.ok and pu.ok):
                print(">> [ERROR] bobot MLP tidak lengkap pada layer", li)
                return
            layers[li].mlp_gate_up_proj = QwenLinear1Bit(
                fuse_u8(pg.w, pg.nbytes, pu.w, pu.nbytes),
                fuse_f32(pg.scales, pg.n_rows * (pg.k_dim // 128), pu.scales, pu.n_rows * (pu.k_dim // 128)),
                fuse_f32(pg.biases, pg.n_rows * (pg.k_dim // 128), pu.biases, pu.n_rows * (pu.k_dim // 128)),
                pg.n_rows + pu.n_rows, pg.k_dim
            )
        var e_dw = find_flex(index, prefix + "mlp.down_proj.weight")
        var e_ds = find_flex(index, prefix + "mlp.down_proj.scales")
        var e_db = find_flex(index, prefix + "mlp.down_proj.biases")
        if e_dw != -1 and e_ds != -1:
            var nb = index.entries[e_dw].nbytes
            var wb = alloc[UInt8](nb)
            var _ = index.read_raw(e_dw, wb, nb)
            var nn = index.dim0(e_dw)
            var kk = nb * 8 // nn
            var sb = read_scales_eff(index, e_ds, nn * (kk // 128))
            var db = read_biases_f32(index, e_db, nn * (kk // 128))
            layers[li].mlp_down_proj = QwenLinear1Bit(wb, sb, db, nn, kk)

    print(">> Bobot termuat:", n_layers, "layer (", n_gdn, "GDN,", n_kv, "attention )")

    # DeviceContext tunggal: bobot diunggah sekali ke global memory T4, bukan
    # per token. Tanpa ini tiap proyeksi membuat-buang context (release race).
    alias T = DType.float16
    var gpu_ctx_ptr = UnsafePointer[DeviceContextGPU, MutAnyOrigin]()

    # Pointer activation workspace VRAM persisten (Full GPU resident)
    var act_hidden_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_x_norm_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_sublayer_out_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_mlp_out_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_proj_raw_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_conv_out_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_q_normed_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_k_normed_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_gdn_out_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_gate_up_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_swiglu_act_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_logits_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_token_out_dev = UnsafePointer[Int32, MutAnyOrigin]()
    var act_stage1_vals_dev = UnsafePointer[Float32, MutAnyOrigin]()
    var act_stage1_idxs_dev = UnsafePointer[Int32, MutAnyOrigin]()
    var fnorm_dev = UnsafePointer[Float32, MutAnyOrigin]()
    var act_attn_scores_dev = UnsafePointer[Float32, MutAnyOrigin]()
    var h_hidden_holder = alloc[DeviceBuffer[T]](1)
    var h_token_out_holder = alloc[DeviceBuffer[DType.int32]](1)

    if use_gpu_matmul():
        gpu_ctx_ptr = gpu_ctx_new()
        for li in range(n_layers):
            layers[li].set_ctx(gpu_ctx_ptr)
        lm_proj.set_ctx(gpu_ctx_ptr)
        print(">> [GPU] DeviceContext tunggal terpasang ke", n_layers, "layer + lm_head")

        # KHQ: jalur KV terkompresi (env BONSAI_KHQ_PATH=<file centroid>).
        # Tanpa fallback: gagal aktivasi = berhenti, bukan diam-diam fp16.
        var khq_path = getenv("BONSAI_KHQ_PATH")
        if khq_path:
            if not khq_activate(gpu_ctx_ptr[], khq_path, 4096):
                raise Error("KHQ gagal aktif (centroid tidak valid): " + khq_path)

        # Inisialisasi VRAM KV Cache untuk seluruh layer Attention (Layer 3, 7, 11, ...)
        for ki in range(n_kv):
            kv_caches[ki].init_device(gpu_ctx_ptr[])

        # Inisialisasi VRAM state untuk seluruh layer GDN (48 layer)
        for gi in range(n_gdn):
            gdn_states[gi].init_device(gpu_ctx_ptr[])

        # Alokasi buffer aktivasi VRAM persisten (seumur proses, < 1 MB VRAM)
        h_hidden_holder.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](D))
        act_hidden_dev = h_hidden_holder[].unsafe_ptr()

        var h_x_norm = alloc[DeviceBuffer[T]](1)
        h_x_norm.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](D))
        act_x_norm_dev = h_x_norm[].unsafe_ptr()

        var h_sub = alloc[DeviceBuffer[T]](1)
        h_sub.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](D))
        act_sublayer_out_dev = h_sub[].unsafe_ptr()

        var h_mlp = alloc[DeviceBuffer[T]](1)
        h_mlp.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](D))
        act_mlp_out_dev = h_mlp[].unsafe_ptr()

        var h_proj = alloc[DeviceBuffer[T]](1)
        h_proj.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](17408))
        act_proj_raw_dev = h_proj[].unsafe_ptr()

        var h_conv = alloc[DeviceBuffer[T]](1)
        h_conv.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](cfg.gdn_conv_dim))
        act_conv_out_dev = h_conv[].unsafe_ptr()

        var h_qn = alloc[DeviceBuffer[T]](1)
        h_qn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](cfg.gdn_num_k_heads * cfg.gdn_head_k_dim))
        act_q_normed_dev = h_qn[].unsafe_ptr()

        var h_kn = alloc[DeviceBuffer[T]](1)
        h_kn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](cfg.gdn_num_k_heads * cfg.gdn_head_k_dim))
        act_k_normed_dev = h_kn[].unsafe_ptr()

        var h_gdn = alloc[DeviceBuffer[T]](1)
        h_gdn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](cfg.gdn_num_v_heads * cfg.gdn_head_v_dim))
        act_gdn_out_dev = h_gdn[].unsafe_ptr()

        var h_gu = alloc[DeviceBuffer[T]](1)
        h_gu.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](2 * cfg.intermediate_size))
        act_gate_up_dev = h_gu[].unsafe_ptr()

        var h_sw = alloc[DeviceBuffer[T]](1)
        h_sw.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](cfg.intermediate_size))
        act_swiglu_act_dev = h_sw[].unsafe_ptr()

        var h_log = alloc[DeviceBuffer[T]](1)
        h_log.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](V))
        act_logits_dev = h_log[].unsafe_ptr()

        h_token_out_holder.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[DType.int32](1))
        act_token_out_dev = h_token_out_holder[].unsafe_ptr()

        var h_s1v = alloc[DeviceBuffer[DType.float32]](1)
        h_s1v.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[DType.float32](256))
        act_stage1_vals_dev = h_s1v[].unsafe_ptr()

        var h_s1i = alloc[DeviceBuffer[DType.int32]](1)
        h_s1i.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[DType.int32](256))
        act_stage1_idxs_dev = h_s1i[].unsafe_ptr()

        var h_fnorm = alloc[DeviceBuffer[DType.float32]](1)
        h_fnorm.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[DType.float32](D))
        gpu_ctx_ptr[].enqueue_copy(h_fnorm[], final_norm_w)
        fnorm_dev = h_fnorm[].unsafe_ptr()

        # Workspace attention scores (H_q * max_seq_len Float32)
        var h_as = alloc[DeviceBuffer[DType.float32]](1)
        h_as.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[DType.float32](cfg.num_attention_heads * max_seq))
        act_attn_scores_dev = h_as[].unsafe_ptr()

        print(">> [GPU] VRAM resident activation buffers siap.")

    # ---------------- 5. Prompt tokens ----------------
    var ptoks = parse_int_list(prompt_spec)
    var prompt_len = ptoks[0] # slot 0 = jumlah elemen
    print(">> Prompt tokens:", prompt_len, "| max_tokens:", max_tokens)

    # ---------------- 6. Generasi greedy ----------------
    # Warmup clock GPU (metodologi benchmark MLX): beberapa iterasi GEMM
    # terbesar (lm_head) menaikkan clock ke keadaan sustain SEBELUM timer —
    # menghapus bias cold-clock pada angka prefill. BONSAI_WARMUP=0 mematikan.
    var warmup_env = getenv("BONSAI_WARMUP")
    if use_gpu_matmul() and not (warmup_env and warmup_env == "0"):
        for _ in range(12):
            lm_proj.forward_device(act_x_norm_dev, act_logits_dev, 1)
        gpu_ctx_ptr[].synchronize()

    var t_all = monotonic()
    var hidden = alloc[Float32](D)
    var generated = alloc[Int](max_tokens + 1)
    var n_generated = 0
    var pos = 0
    var next_tok = 0
    var logits = alloc[Float32](V)

    if use_gpu_matmul():
        var next_tok_host = alloc[Int32](1)

        # Upload embedding 1-bit ke VRAM SEKALI (packed + skala + bias):
        # lookup per token jalan di GPU (embed_lookup_1bit_sm75) — menghapus
        # dequant CPU + cast skalar + H2D PCIe pada SETIAP token.
        var embed_w_dev_buf = alloc[DeviceBuffer[DType.uint8]](1)
        embed_w_dev_buf.init_pointee_move(
            gpu_ctx_ptr[].enqueue_create_buffer[DType.uint8](index.entries[e_emb].nbytes)
        )
        gpu_ctx_ptr[].enqueue_copy(embed_w_dev_buf[], embed_w)
        var embed_s_dev_buf = alloc[DeviceBuffer[DType.float32]](1)
        embed_s_dev_buf.init_pointee_move(
            gpu_ctx_ptr[].enqueue_create_buffer[DType.float32](V_emb * (D_real // 128))
        )
        gpu_ctx_ptr[].enqueue_copy(embed_s_dev_buf[], embed_s)
        var embed_b_dev_buf = alloc[DeviceBuffer[DType.float32]](1)
        embed_b_dev_buf.init_pointee_move(
            gpu_ctx_ptr[].enqueue_create_buffer[DType.float32](V_emb * (D_real // 128))
        )
        gpu_ctx_ptr[].enqueue_copy(embed_b_dev_buf[], embed_b)
        var embed_w_dev = embed_w_dev_buf[].unsafe_ptr()
        var embed_s_dev = embed_s_dev_buf[].unsafe_ptr()
        var embed_b_dev = embed_b_dev_buf[].unsafe_ptr()

        # Timer prefill mulai SETELAH upload embedding (~238 MB via PCIe) —
        # itu biaya setup one-time, bukan beban latensi prompt.
        var t_prefill_start = monotonic()

        # prefill (Full GPU resident)
        # Jalur BATCHED (default): prompt diproses per chunk M-token via kernel
        # qmm WMMA v2 (tensor core) — bobot 2.8 GB dibaca SEKALI per chunk,
        # bukan M kali seperti GEMV per-token. Fallback per-token otomatis bila
        # .so tidak memiliki simbol qmm atau env BONSAI_PREFILL_PER_TOKEN=1.
        var force_per_token = getenv("BONSAI_PREFILL_PER_TOKEN")
        var batched_prefill = layers[0].prefill_ffi_ready() and not (
            force_per_token and force_per_token[0] == "1"
        )
        if batched_prefill:
            alias PF_CHUNK = 256
            var chunk_len = min(PF_CHUNK, prompt_len)

            # Buffer prefill M-token (fp16, RAII hidup selama blok ini)
            var h_pf_hidden = alloc[DeviceBuffer[T]](1)
            h_pf_hidden.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * D))
            var pf_hidden = h_pf_hidden[].unsafe_ptr()
            var h_pf_xn = alloc[DeviceBuffer[T]](1)
            h_pf_xn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * D))
            var pf_xn = h_pf_xn[].unsafe_ptr()
            var h_pf_sub = alloc[DeviceBuffer[T]](1)
            h_pf_sub.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * D))
            var pf_sub = h_pf_sub[].unsafe_ptr()
            var h_pf_mlp = alloc[DeviceBuffer[T]](1)
            h_pf_mlp.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * D))
            var pf_mlp = h_pf_mlp[].unsafe_ptr()
            var h_pf_proj = alloc[DeviceBuffer[T]](1)
            h_pf_proj.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * 17408))
            var pf_proj = h_pf_proj[].unsafe_ptr()
            var h_pf_conv = alloc[DeviceBuffer[T]](1)
            h_pf_conv.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * cfg.gdn_conv_dim))
            var pf_conv = h_pf_conv[].unsafe_ptr()
            var pf_qk_dim = cfg.gdn_num_k_heads * cfg.gdn_head_k_dim
            var h_pf_qn = alloc[DeviceBuffer[T]](1)
            h_pf_qn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * pf_qk_dim))
            var pf_qn = h_pf_qn[].unsafe_ptr()
            var h_pf_kn = alloc[DeviceBuffer[T]](1)
            h_pf_kn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * pf_qk_dim))
            var pf_kn = h_pf_kn[].unsafe_ptr()
            var pf_v_dim = cfg.gdn_num_v_heads * cfg.gdn_head_v_dim
            var h_pf_gdn = alloc[DeviceBuffer[T]](1)
            h_pf_gdn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * pf_v_dim))
            var pf_gdn = h_pf_gdn[].unsafe_ptr()
            var h_pf_gu = alloc[DeviceBuffer[T]](1)
            h_pf_gu.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * 2 * cfg.intermediate_size))
            var pf_gu = h_pf_gu[].unsafe_ptr()
            var h_pf_sw = alloc[DeviceBuffer[T]](1)
            h_pf_sw.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * cfg.intermediate_size))
            var pf_sw = h_pf_sw[].unsafe_ptr()

            var pos_pf = 0
            while pos_pf < prompt_len:
                var m = min(PF_CHUNK, prompt_len - pos_pf)
                # Embed lookup per baris (M row di pf_hidden)
                for t in range(m):
                    embed_lookup_1bit_sm75_launch_on[T](
                        gpu_ctx_ptr[], embed_w_dev, embed_s_dev, embed_b_dev,
                        pf_hidden.offset(t * D), ptoks[pos_pf + t + 1], D
                    )
                for li in range(n_layers):
                    if layers[li].is_linear:
                        layers[li].forward_prefill_gpu(
                            pf_hidden, pf_xn, pf_sub, pf_mlp,
                            pf_proj, pf_conv, pf_qn, pf_kn, pf_gdn,
                            pf_gu, pf_sw, act_attn_scores_dev,
                            gdn_states[gdn_idx[li]], kv_caches[0], pos_pf, m
                        )
                    else:
                        layers[li].forward_prefill_gpu(
                            pf_hidden, pf_xn, pf_sub, pf_mlp,
                            pf_proj, pf_conv, pf_qn, pf_kn, pf_gdn,
                            pf_gu, pf_sw, act_attn_scores_dev,
                            gdn_states[0], kv_caches[kv_idx[li]], pos_pf, m
                        )
                pos_pf += m
            pos = pos_pf

            # Transplantasi baris terakhir prefill -> buffer decode (D2D)
            copy_vec_sm75_launch_on[T](
                gpu_ctx_ptr[], act_hidden_dev,
                pf_hidden.offset((prompt_len - 1) % chunk_len * D), D
            )
            gpu_ctx_ptr[].synchronize()

            # Norm final + lm_head + argmax hanya pada baris terakhir
            rmsnorm_sm75_launch_on[T](
                gpu_ctx_ptr[], act_hidden_dev, act_x_norm_dev,
                fnorm_dev, True, D, cfg.rms_norm_eps
            )
            lm_proj.forward_device(act_x_norm_dev, act_logits_dev, 1)

            # Diagnostik near-tie di batas prefill (BONSAI_DUMP_TOP2=1)
            var dumpv = getenv("BONSAI_DUMP_TOP2")
            if dumpv and dumpv[0] == "1":
                dump_top2_prefill(gpu_ctx_ptr, act_logits_dev, V)

            argmax_sm75_launch_on[T](
                gpu_ctx_ptr[], act_logits_dev, act_stage1_vals_dev, act_stage1_idxs_dev,
                act_token_out_dev, V
            )
            gpu_ctx_ptr[].enqueue_copy(next_tok_host, h_token_out_holder[])
            gpu_ctx_ptr[].synchronize()
            next_tok = Int(next_tok_host[0])
            # Buffer pf_* dibebaskan otomatis oleh RAII di akhir blok.
        else:
            # Fallback per-token (GEMV decode per baris; hasil identik)
            for t in range(prompt_len):
                var cur_tok = ptoks[t + 1]
                embed_lookup_1bit_sm75_launch_on[T](
                    gpu_ctx_ptr[], embed_w_dev, embed_s_dev, embed_b_dev,
                    h_hidden_holder[].unsafe_ptr(), cur_tok, D
                )

                for li in range(n_layers):
                    if layers[li].is_linear:
                        var gi = gdn_idx[li]
                        layers[li].forward_gpu(
                            act_hidden_dev, act_x_norm_dev, act_sublayer_out_dev, act_mlp_out_dev,
                            act_proj_raw_dev, act_conv_out_dev, act_q_normed_dev, act_k_normed_dev,
                            act_gdn_out_dev, act_gate_up_dev, act_swiglu_act_dev,
                            act_attn_scores_dev,
                            gdn_states[gi], kv_caches[0], pos
                        )
                    else:
                        var ki = kv_idx[li]
                        layers[li].forward_gpu(
                            act_hidden_dev, act_x_norm_dev, act_sublayer_out_dev, act_mlp_out_dev,
                            act_proj_raw_dev, act_conv_out_dev, act_q_normed_dev, act_k_normed_dev,
                            act_gdn_out_dev, act_gate_up_dev, act_swiglu_act_dev,
                            act_attn_scores_dev,
                            gdn_states[0], kv_caches[ki], pos
                        )

                # LM head + argmax hanya untuk token TERAKHIR prompt (yang
                # menghasilkan prediksi token generasi pertama). Untuk token
                # intermediate next_tok dibuang — cur_tok diambil dari prompt —
                # jadi lewati proyeksi V=248320 + reduksi argmax + sync host.
                # State GDN/KV tetap ter-update asinkron via stream yang sama.
                if t == prompt_len - 1:
                    rmsnorm_sm75_launch_on[T](
                        gpu_ctx_ptr[], act_hidden_dev, act_x_norm_dev,
                        fnorm_dev, True, D, cfg.rms_norm_eps
                    )
                    lm_proj.forward_device(act_x_norm_dev, act_logits_dev, 1)

                    # Diagnostik near-tie di batas prefill (BONSAI_DUMP_TOP2=1)
                    var dumpv2 = getenv("BONSAI_DUMP_TOP2")
                    if dumpv2 and dumpv2[0] == "1":
                        dump_top2_prefill(gpu_ctx_ptr, act_logits_dev, V)

                    argmax_sm75_launch_on[T](
                        gpu_ctx_ptr[], act_logits_dev, act_stage1_vals_dev, act_stage1_idxs_dev,
                        act_token_out_dev, V
                    )
                    gpu_ctx_ptr[].enqueue_copy(next_tok_host, h_token_out_holder[])
                    gpu_ctx_ptr[].synchronize()
                    next_tok = Int(next_tok_host[0])
                pos += 1
        generated[n_generated] = next_tok
        n_generated += 1
        print(">> [GEN] token id:", next_tok)
        var t_decode = monotonic()

        # decode (Full GPU resident)
        # Profil per-subsystem (BONSAI_PROFILE=1): sinkronisasi per layer agar
        # waktu GPU ter-atribusi akurat ke GDN / Attention / LM head.
        var prof_v = getenv("BONSAI_PROFILE")
        var prof = prof_v and prof_v[0] == "1"
        var acc_gdn: Int = 0
        var acc_attn: Int = 0
        var acc_lm: Int = 0
        for step in range(max_tokens - 1):
            var t0 = monotonic()
            embed_lookup_1bit_sm75_launch_on[T](
                gpu_ctx_ptr[], embed_w_dev, embed_s_dev, embed_b_dev,
                h_hidden_holder[].unsafe_ptr(), next_tok, D
            )

            for li in range(n_layers):
                var tl = monotonic()
                # Fusi residual2+pre-norm: weight norm layer berikutnya
                # (layer terakhir: final norm sebelum lm_head).
                var nnw = (
                    layers[li + 1].input_layernorm_w_dev
                    if li + 1 < n_layers else fnorm_dev
                )
                if layers[li].is_linear:
                    var gi = gdn_idx[li]
                    layers[li].forward_gpu(
                        act_hidden_dev, act_x_norm_dev, act_sublayer_out_dev, act_mlp_out_dev,
                        act_proj_raw_dev, act_conv_out_dev, act_q_normed_dev, act_k_normed_dev,
                        act_gdn_out_dev, act_gate_up_dev, act_swiglu_act_dev,
                        act_attn_scores_dev,
                        gdn_states[gi], kv_caches[0], pos,
                        nnw, li > 0
                    )
                    if prof:
                        gpu_ctx_ptr[].synchronize()
                    acc_gdn += monotonic() - tl
                else:
                    var ki = kv_idx[li]
                    layers[li].forward_gpu(
                        act_hidden_dev, act_x_norm_dev, act_sublayer_out_dev, act_mlp_out_dev,
                        act_proj_raw_dev, act_conv_out_dev, act_q_normed_dev, act_k_normed_dev,
                        act_gdn_out_dev, act_gate_up_dev, act_swiglu_act_dev,
                        act_attn_scores_dev,
                        gdn_states[0], kv_caches[ki], pos,
                        nnw, li > 0
                    )
                    if prof:
                        gpu_ctx_ptr[].synchronize()
                    acc_attn += monotonic() - tl

            if prof:
                gpu_ctx_ptr[].synchronize()
            var tlm = monotonic()

            # Final norm dilewati bila fusi aktif: residual-2 layer terakhir
            # sudah menulis act_x_norm_dev dgn fnorm_dev (bit-exact identik).
            var no_fuse = getenv("BONSAI_NO_FUSE")
            if not (no_fuse and no_fuse == "1"):
                pass
            else:
                rmsnorm_sm75_launch_on[T](
                    gpu_ctx_ptr[], act_hidden_dev, act_x_norm_dev,
                    fnorm_dev, True, D, cfg.rms_norm_eps
                )
            lm_proj.forward_device(act_x_norm_dev, act_logits_dev, 1)
            argmax_sm75_launch_on[T](
                gpu_ctx_ptr[], act_logits_dev, act_stage1_vals_dev, act_stage1_idxs_dev,
                act_token_out_dev, V
            )
            gpu_ctx_ptr[].enqueue_copy(next_tok_host, h_token_out_holder[])
            gpu_ctx_ptr[].synchronize()
            acc_lm += monotonic() - tlm
            next_tok = Int(next_tok_host[0])
            pos += 1
            generated[n_generated] = next_tok
            n_generated += 1
            var dt = monotonic() - t0
            var ms = Float64(dt) / 1e6
            var tps = 1000.0 / ms if ms > 0.0 else 0.0
            print(">> [GEN] token id:", next_tok, "|", ms, "ms |", tps, "tok/s")

        next_tok_host.free()

        var total_ms = Float64(monotonic() - t_all) / 1e6
        var dec_ms = Float64(monotonic() - t_decode) / 1e6
        var prefill_ms = Float64(t_decode - t_prefill_start) / 1e6
        var n_dec = max_tokens - 1
        if n_dec > 0:
            print(">> [PERF] prefill", prompt_len, "token |", prefill_ms, "ms |",
                  prefill_ms / Float64(prompt_len), "ms/token |",
                  Float64(prompt_len) * 1000.0 / prefill_ms if prefill_ms > 0.0 else 0.0,
                  "tok/s | decode", n_dec, "token")
            print(">> [PERF] rata-rata decode:", dec_ms / n_dec, "ms/token |",
                  Float64(n_dec) * 1000.0 / dec_ms if dec_ms > 0.0 else 0.0, "tok/s")
            print(">> [PROF/SPLIT] per token -> GDN:",
                  Float64(acc_gdn) / 1e6 / n_dec, "ms | ATTN:",
                  Float64(acc_attn) / 1e6 / n_dec, "ms | LM_HEAD+argmax:",
                  Float64(acc_lm) / 1e6 / n_dec, "ms")
        print(">> [PERF] total:", total_ms, "ms")
        print(">> Selesai:", n_generated, "token di-generate (greedy).")
        if khq_dir:
            khq_dump_flush()
    else:
        # FALLBACK DIHAPUS: jalur host-sim CPU tidak lagi dipakai produksi.
        # GPU wajib — tanpa BONSAI_USE_GPU program error, bukan diam-diam CPU.
        raise Error(
            "CPU fallback dilarang: jalur host-sim telah dihapus. "
            "Set BONSAI_USE_GPU=1 agar inferensi berjalan di NVIDIA T4."
        )

    hidden.free()
    logits.free()
    generated.free()
