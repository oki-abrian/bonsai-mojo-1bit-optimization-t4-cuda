# ===----------------------------------------------------------------------=== #
# Module: src/loader.py
# Purpose: SafeTensors Weight Loader & Memory Mapper untuk Model Qwen/Bonsai 27B
#          Membaca file safetensors sharded langsung ke pointer memori tanpa duplikasi.
# ===----------------------------------------------------------------------=== #

import os
import json
import struct
import mmap
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

class SafeTensorsLoader:
    """
    Loader SafeTensors berkinerja tinggi dengan memory-mapping (mmap):
    Mendukung format single-file maupun multi-shard (model-00001-of-0000X.safetensors).
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.index_file = self.model_dir / "model.safetensors.index.json"
        self.single_file = self.model_dir / "model.safetensors"
        self.tensor_map: Dict[str, str] = {} # Nama tensor -> nama file shard
        self.open_mmaps: Dict[str, mmap.mmap] = {}
        self.tensor_headers: Dict[str, Dict[str, Any]] = {}
        self._initialize()

    def _initialize(self):
        """Membaca index shard atau file tunggal safetensors."""
        if self.index_file.exists():
            with open(self.index_file, "r") as f:
                index_data = json.load(f)
            self.tensor_map = index_data.get("weight_map", {})
            print(f">> [LOADER] Membaca indeks multi-shard: {len(self.tensor_map)} tensor terdaftar.")
        elif self.single_file.exists():
            print(">> [LOADER] Menggunakan file tunggal: model.safetensors")
            # Parse header untuk mendapatkan daftar tensor
            self._load_shard_header(str(self.single_file))
            for t_name in self.tensor_headers.keys():
                self.tensor_map[t_name] = "model.safetensors"
        else:
            # Cari seluruh *.safetensors di direktori
            shards = list(self.model_dir.glob("*.safetensors"))
            if shards:
                print(f">> [LOADER] Ditemukan {len(shards)} shard safetensors di {self.model_dir}.")
                for shard in shards:
                    self._load_shard_header(str(shard))
                    shard_rel = shard.name
                    # Parse header shard ini
                    with open(shard, "rb") as f:
                        header_size = struct.unpack("<Q", f.read(8))[0]
                        header_json = json.loads(f.read(header_size).decode("utf-8"))
                        for k in header_json.keys():
                            if k != "__metadata__":
                                self.tensor_map[k] = shard_rel
            else:
                print(f">> [WARN] Tidak ada file .safetensors ditemukan di {self.model_dir}")

    def _get_mmap(self, shard_name: str) -> Tuple[mmap.mmap, int]:
        """Mendapatkan objek memory-map dan offset data dari file shard."""
        if shard_name not in self.open_mmaps:
            file_path = self.model_dir / shard_name
            f = open(file_path, "rb")
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes.decode("utf-8"))
            data_offset = 8 + header_size
            
            # Map seluruh file ke memori
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.open_mmaps[shard_name] = (mm, data_offset, header)

        mm, data_offset, header = self.open_mmaps[shard_name]
        return mm, data_offset, header

    def get_tensor(self, tensor_name: str) -> Optional[Dict[str, Any]]:
        """
        Mengambil pointer memori buffer mentah, tipe data, dan bentuk tensor:
        Mengembalikan dict: { 'data_ptr': buffer, 'shape': list, 'dtype': str }
        """
        if tensor_name not in self.tensor_map:
            return None

        shard_name = self.tensor_map[tensor_name]
        mm, data_offset, header = self._get_mmap(shard_name)

        if tensor_name not in header:
            return None

        t_meta = header[tensor_name]
        dtype = t_meta["dtype"]
        shape = t_meta["shape"]
        offsets = t_meta["data_offsets"]

        start = data_offset + offsets[0]
        end = data_offset + offsets[1]
        raw_buffer = memoryview(mm)[start:end]

        return {
            "name": tensor_name,
            "shape": shape,
            "dtype": dtype,
            "buffer": raw_buffer,
            "nbytes": offsets[1] - offsets[0]
        }

    def load_layer_weights(self, layer_idx: int) -> Dict[str, Any]:
        """
        Memuat seluruh tensor terkuantisasi dan FP32 untuk 1 layer transformer:
        - GDN: in_proj_all.weight, in_proj_all.scales, out_proj, conv1d
        - Attention: q_proj, k_proj, v_proj, o_proj
        - MLP: gate_up_proj, down_proj
        - RMSNorm: input_layernorm, post_attention_layernorm
        """
        prefix = f"model.layers.{layer_idx}."
        weights = {}

        expected_tensors = [
            # Normalisasi
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            # GDN / Attention
            "linear_attn.in_proj_all.weight",
            "linear_attn.in_proj_all.scales",
            "linear_attn.out_proj.weight",
            "linear_attn.out_proj.scales",
            "linear_attn.conv1d.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.scales",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.scales",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.scales",
            "self_attn.o_proj.weight",
            "self_attn.o_proj.scales",
            # MLP SwiGLU
            "mlp.gate_up_proj.weight",
            "mlp.gate_up_proj.scales",
            "mlp.down_proj.weight",
            "mlp.down_proj.scales"
        ]

        for suffix in expected_tensors:
            full_name = prefix + suffix
            t = self.get_tensor(full_name)
            if t is not None:
                weights[suffix] = t

        # -----------------------------------------------------------------
        # PAKET C: In-Place Weight Fusion (Jika Checkpoint Menyimpan Terpisah)
        # -----------------------------------------------------------------
        # 1. Fusi Linear MLP: gate_proj + up_proj -> gate_up_proj (1 QMV Call)
        if "mlp.gate_up_proj.weight" not in weights:
            g_w = self.get_tensor(prefix + "mlp.gate_proj.weight")
            u_w = self.get_tensor(prefix + "mlp.up_proj.weight")
            g_s = self.get_tensor(prefix + "mlp.gate_proj.scales")
            u_s = self.get_tensor(prefix + "mlp.up_proj.scales")
            if g_w and u_w and g_s and u_s:
                fused_w_bytes = bytes(g_w["buffer"]) + bytes(u_w["buffer"])
                fused_s_bytes = bytes(g_s["buffer"]) + bytes(u_s["buffer"])
                weights["mlp.gate_up_proj.weight"] = {
                    "name": prefix + "mlp.gate_up_proj.weight",
                    "shape": [g_w["shape"][0] + u_w["shape"][0], g_w["shape"][1]],
                    "dtype": g_w["dtype"],
                    "buffer": memoryview(fused_w_bytes),
                    "nbytes": len(fused_w_bytes)
                }
                weights["mlp.gate_up_proj.scales"] = {
                    "name": prefix + "mlp.gate_up_proj.scales",
                    "shape": [g_s["shape"][0] + u_s["shape"][0], g_s["shape"][1]],
                    "dtype": g_s["dtype"],
                    "buffer": memoryview(fused_s_bytes),
                    "nbytes": len(fused_s_bytes)
                }

        # 2. Fusi Linear GDN: qkv + z + b + a -> in_proj_all (1 QMV Call)
        if "linear_attn.in_proj_all.weight" not in weights:
            qkv_w = self.get_tensor(prefix + "linear_attn.in_proj_qkv.weight")
            z_w = self.get_tensor(prefix + "linear_attn.in_proj_z.weight")
            b_w = self.get_tensor(prefix + "linear_attn.in_proj_b.weight")
            a_w = self.get_tensor(prefix + "linear_attn.in_proj_a.weight")
            qkv_s = self.get_tensor(prefix + "linear_attn.in_proj_qkv.scales")
            z_s = self.get_tensor(prefix + "linear_attn.in_proj_z.scales")
            b_s = self.get_tensor(prefix + "linear_attn.in_proj_b.scales")
            a_s = self.get_tensor(prefix + "linear_attn.in_proj_a.scales")
            if qkv_w and z_w and b_w and a_w and qkv_s and z_s and b_s and a_s:
                fused_w_bytes = bytes(qkv_w["buffer"]) + bytes(z_w["buffer"]) + bytes(b_w["buffer"]) + bytes(a_w["buffer"])
                fused_s_bytes = bytes(qkv_s["buffer"]) + bytes(z_s["buffer"]) + bytes(b_s["buffer"]) + bytes(a_s["buffer"])
                total_n = qkv_w["shape"][0] + z_w["shape"][0] + b_w["shape"][0] + a_w["shape"][0]
                weights["linear_attn.in_proj_all.weight"] = {
                    "name": prefix + "linear_attn.in_proj_all.weight",
                    "shape": [total_n, qkv_w["shape"][1]],
                    "dtype": qkv_w["dtype"],
                    "buffer": memoryview(fused_w_bytes),
                    "nbytes": len(fused_w_bytes)
                }
                weights["linear_attn.in_proj_all.scales"] = {
                    "name": prefix + "linear_attn.in_proj_all.scales",
                    "shape": [total_n, qkv_s["shape"][1]],
                    "dtype": qkv_s["dtype"],
                    "buffer": memoryview(fused_s_bytes),
                    "nbytes": len(fused_s_bytes)
                }

        return weights

    def close(self):
        """Menutup semua file memory map."""
        for mm, _, _ in self.open_mmaps.values():
            mm.close()
        self.open_mmaps.clear()

    def __del__(self):
        self.close()
