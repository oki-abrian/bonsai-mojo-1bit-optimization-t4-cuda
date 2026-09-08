# ===----------------------------------------------------------------------=== #
# Module: safetensors.mojo
# Purpose: Pembaca safetensors native Mojo (tanpa Python):
#          - indeks multi-shard via model.safetensors.index.json (jsonlite)
#          - fallback file tunggal model.safetensors
#          - konversi F16/BF16 -> F32 (eksak) saat load; U8/U32 disalin
#            mentah (bobot terpaket 1-bit), helper fusion paritas loader.py
# Catatan port (lihat NOTES_API_IO.md / NOTES_API_BITCAST.md):
#          - paket `libc` sudah dihapus; file I/O memakai `io.file`:
#            open(path, mode) -> FileHandle, seek, read(Span), close
#          - `bitcast` dari modul `memory` (bitwidth sumber = target)
#          - F16 -> F32 memakai konversi native Float32(h) (eksak semua nilai)
#          - bentuk import `io.file` vs prefix `std.io.file` diverifikasi pada
#            kompilasi pertama; open() diasumsikan raising (dibungkus raises)
# ===----------------------------------------------------------------------=== #

from io.file import FileHandle, open
from memory import UnsafePointer, bitcast, alloc
from .jsonlite import JsonDoc, JSON_OBJ, JSON_ARR, JSON_STR

# Kode dtype
alias DT_F32: UInt8 = 0
alias DT_F16: UInt8 = 1
alias DT_BF16: UInt8 = 2
alias DT_U8: UInt8 = 3
alias DT_U32: UInt8 = 4

fn _slurp_file(path: String, mut out_len: Int) raises -> UnsafePointer[UInt8, MutAnyOrigin]:
    """Baca SELURUH file ke buffer heap sampai EOF.

    PENTING: model.safetensors.index.json adalah JSON POLOS — BUKAN kontainer
    safetensors ber-header panjang 8-byte. Membaca 8 byte pertamanya sebagai
    panjang menghasilkan angka raksasa (alloc gagal -> abort runtime)."""
    var f = open(path, "r")
    var cap = 1 << 20
    var buf = alloc[UInt8](cap)
    var total = 0
    var done = False
    while not done:
        if total == cap:
            var nb = alloc[UInt8](cap * 2)
            for i in range(cap):
                nb[i] = buf[i]
            buf.free()
            buf = nb
            cap *= 2
        var n = f.read(Span[UInt8, MutAnyOrigin](ptr=buf + total, length=cap - total))
        if n <= 0:
            done = True
        else:
            total += Int(n)
    f.close()
    out_len = total
    return buf

@fieldwise_init
struct STEntry(Copyable, Movable, ImplicitlyCopyable):
    """Metadata satu tensor: lokasi di shard + bentuk + dtype."""
    var name_off: Int
    var name_len: Int
    var shard_idx: Int
    var start: Int
    var nbytes: Int
    var d0: Int
    var d1: Int
    var d2: Int
    var dtype: UInt8

struct SafeTensorsIndex:
    """
    Indeks tensor safetensors multi-shard:
    - nama tensor disimpan di arena `names` (tanpa nul, dipisah rapat)
    - tiap tensor punya STEntry (shard, start, nbytes, shape, dtype)
    - path shard disimpan berurutan di arena `shard_paths`
    File shard dibuka per-pembacaan tensor (buka -> seek -> read -> tutup).
    """
    var names: UnsafePointer[UInt8, MutAnyOrigin]
    var n_names: Int
    var cap_names: Int
    var entries: UnsafePointer[STEntry, MutAnyOrigin]
    var n_entries: Int
    var cap_entries: Int
    var shard_paths: UnsafePointer[UInt8, MutAnyOrigin]
    var n_shard_bytes: Int
    var shard_offs: UnsafePointer[Int, MutAnyOrigin]
    var shard_lens: UnsafePointer[Int, MutAnyOrigin]
    var shard_header_lens: UnsafePointer[Int, MutAnyOrigin]
    var n_shards: Int
    var ok: Bool

    fn __init__(out self):
        self.names = UnsafePointer[UInt8, MutAnyOrigin]()
        self.n_names = 0
        self.cap_names = 0
        self.entries = UnsafePointer[STEntry, MutAnyOrigin]()
        self.n_entries = 0
        self.cap_entries = 0
        self.shard_paths = UnsafePointer[UInt8, MutAnyOrigin]()
        self.n_shard_bytes = 0
        self.shard_offs = UnsafePointer[Int, MutAnyOrigin]()
        self.shard_lens = UnsafePointer[Int, MutAnyOrigin]()
        self.shard_header_lens = UnsafePointer[Int, MutAnyOrigin]()
        self.n_shards = 0
        self.ok = False

    fn free(self):
        if self.cap_names > 0:
            self.names.free()
        if self.cap_entries > 0:
            self.entries.free()
        if self.n_shard_bytes > 0:
            self.shard_paths.free()
        if self.n_shards > 0:
            self.shard_offs.free()
            self.shard_lens.free()
            self.shard_header_lens.free()

    # ---------------- arena internal ----------------
    fn _add_name_from_doc(mut self, doc: JsonDoc, node: Int) -> Int:
        var n = doc.str_len_of(node)
        if self.n_names + n > self.cap_names:
            var new_cap = 65536 if self.cap_names == 0 else self.cap_names * 2
            while self.n_names + n > new_cap:
                new_cap *= 2
            var fresh = alloc[UInt8](new_cap)
            for i in range(self.n_names):
                fresh[i] = self.names[i]
            if self.cap_names > 0:
                self.names.free()
            self.names = fresh
            self.cap_names = new_cap
        var off = self.n_names
        for i in range(n):
            self.names[off + i] = doc.sbuf[doc.nodes[node].str_off + i]
        self.n_names += n
        return off

    fn _add_entry(mut self, e: STEntry) -> Int:
        if self.n_entries == self.cap_entries:
            var new_cap = 1024 if self.cap_entries == 0 else self.cap_entries * 2
            var fresh = alloc[STEntry](new_cap)
            for i in range(self.n_entries):
                fresh[i] = self.entries[i]
            if self.cap_entries > 0:
                self.entries.free()
            self.entries = fresh
            self.cap_entries = new_cap
        var idx = self.n_entries
        self.entries[idx] = e
        self.n_entries += 1
        return idx

    fn _add_shard_path(mut self, s: String) -> Int:
        var n = len(s)
        # +1: byte nul terminator (wajib — dipakai unsafe_from_utf8_ptr)
        var fresh = alloc[UInt8](self.n_shard_bytes + n + 1)
        for i in range(self.n_shard_bytes):
            fresh[i] = self.shard_paths[i]
        var bytes = s.as_bytes()
        for i in range(n):
            fresh[self.n_shard_bytes + i] = bytes[i]
        if self.n_shard_bytes > 0:
            self.shard_paths.free()
        self.shard_paths = fresh
        var off = self.n_shard_bytes
        self.n_shard_bytes += n
        var fresh_off = alloc[Int](self.n_shards + 1)
        var fresh_len = alloc[Int](self.n_shards + 1)
        var fresh_hl = alloc[Int](self.n_shards + 1)
        for i in range(self.n_shards):
            fresh_off[i] = self.shard_offs[i]
            fresh_len[i] = self.shard_lens[i]
            fresh_hl[i] = self.shard_header_lens[i]
        fresh_off[self.n_shards] = off
        fresh_len[self.n_shards] = n
        fresh_hl[self.n_shards] = 0
        if self.n_shards > 0:
            self.shard_offs.free()
            self.shard_lens.free()
            self.shard_header_lens.free()
        self.shard_offs = fresh_off
        self.shard_lens = fresh_len
        self.shard_header_lens = fresh_hl
        self.n_shards += 1
        return self.n_shards - 1

    fn _shard_path_is(self, shard_idx: Int, s: String) -> Bool:
        var off = self.shard_offs[shard_idx]
        var n = self.shard_lens[shard_idx]
        var bytes = s.as_bytes()
        if len(bytes) != n:
            return False
        for i in range(n):
            if self.shard_paths[off + i] != bytes[i]:
                return False
        return True

    # ---------------- parse satu shard ----------------
    fn _dtype_code(self, doc: JsonDoc, node: Int) -> UInt8:
        var n = doc.str_len_of(node)
        var buf = alloc[UInt8](n if n > 0 else 1)
        var _ = doc.str_copy(node, buf, n)
        var code = DT_U8
        if n == 3 and buf[0] == UInt8(ord("F")) and buf[1] == UInt8(ord("3")) and buf[2] == UInt8(ord("2")):
            code = DT_F32
        elif n == 3 and buf[0] == UInt8(ord("F")) and buf[1] == UInt8(ord("1")) and buf[2] == UInt8(ord("6")):
            code = DT_F16
        elif n == 4 and buf[0] == UInt8(ord("B")) and buf[1] == UInt8(ord("F")) and buf[2] == UInt8(ord("1")) and buf[3] == UInt8(ord("6")):
            code = DT_BF16
        elif n == 3 and buf[0] == UInt8(ord("U")) and buf[1] == UInt8(ord("3")) and buf[2] == UInt8(ord("2")):
            code = DT_U32
        buf.free()
        return code

    fn _scan_shard(mut self, path: String, shard_idx: Int) raises:
        """Buka satu shard: baca header (8 byte panjang LE + JSON), daftarkan
        seluruh tensor di dalamnya ke indeks. Panjang header disimpan per
        shard karena offset tensor relatif terhadap data section."""
        var handle = open(path, "r")
        var head8 = alloc[UInt8](8)
        var got = handle.read(Span[UInt8, MutAnyOrigin](ptr=head8, length=8))
        if got != 8:
            head8.free()
            handle.close()
            return
        var header_len = 0
        for b in range(8):
            header_len += Int(head8[b]) << (b * 8)
        self.shard_header_lens[shard_idx] = header_len
        head8.free()
        if header_len <= 0:
            handle.close()
            return
        var hbuf = alloc[UInt8](header_len)
        var got2 = handle.read(Span[UInt8, MutAnyOrigin](ptr=hbuf, length=header_len))
        handle.close()
        if got2 != header_len:
            hbuf.free()
            return

        var doc = JsonDoc()
        if not doc.parse_bytes(hbuf, header_len):
            hbuf.free()
            doc.free()
            return
        hbuf.free()

        # iterasi seluruh pasangan root: key = nama tensor, value = metadata
        var pair = doc.first_child(doc.root())
        while pair != -1:
            var key_len = doc.str_len_of(pair)
            var name_off = self._add_name_from_doc(doc, pair)
            var meta = doc.first_child(pair)
            if meta != -1 and doc.kind(meta) == JSON_OBJ:
                var e = STEntry(
                    name_off=name_off, name_len=key_len, shard_idx=shard_idx,
                    start=0, nbytes=0, d0=1, d1=1, d2=1, dtype=DT_U8
                )
                var dnode = doc.obj_get(meta, "dtype")
                if dnode != -1 and doc.kind(dnode) == JSON_STR:
                    e.dtype = self._dtype_code(doc, dnode)
                var offs = doc.obj_get(meta, "data_offsets")
                if offs != -1 and doc.kind(offs) == JSON_ARR:
                    var s_node = doc.child_at(offs, 0)
                    var e_node = doc.child_at(offs, 1)
                    if s_node != -1:
                        e.start = doc.as_int(s_node)
                    if e_node != -1:
                        e.nbytes = doc.as_int(e_node) - e.start
                var shape = doc.obj_get(meta, "shape")
                if shape != -1 and doc.kind(shape) == JSON_ARR:
                    var sn = doc.arr_len(shape)
                    if sn > 0:
                        e.d0 = doc.as_int(doc.child_at(shape, 0))
                    if sn > 1:
                        e.d1 = doc.as_int(doc.child_at(shape, 1))
                    if sn > 2:
                        e.d2 = doc.as_int(doc.child_at(shape, 2))
                var _ = self._add_entry(e)
            pair = doc.next_sibling(pair)
        doc.free()

    # ---------------- API publik ----------------
    fn open_dir(mut self, model_dir: String) raises:
        """Buka model: coba index multi-shard dulu, lalu file tunggal."""
        # 1. index multi-shard
        var idx_path = model_dir + "/" + "model.safetensors.index.json"
        var index_doc = JsonDoc()
        var have_index = False
        try:
            # index.json = JSON polos: baca utuh sampai EOF (bukan header 8B)
            var ilen = 0
            var ibuf = _slurp_file(idx_path, ilen)
            if ilen > 0 and index_doc.parse_bytes(ibuf, ilen):
                have_index = True
            ibuf.free()
        except:
            have_index = False

        if have_index:
            # kumpulkan nama shard unik dari weight_map, scan tiap shard
            var wm = index_doc.obj_get(index_doc.root(), "weight_map")
            if wm != -1 and index_doc.kind(wm) == JSON_OBJ:
                var pair = index_doc.first_child(wm)
                while pair != -1:
                    var shard_node = index_doc.first_child(pair)
                    if shard_node != -1 and index_doc.kind(shard_node) == JSON_STR:
                        var slen = index_doc.str_len_of(shard_node)
                        var sbuf2 = alloc[UInt8](slen + 1)
                        var _ = index_doc.str_copy(shard_node, sbuf2, slen)
                        sbuf2[slen] = 0
                        var shard_name = String(unsafe_from_utf8_ptr=sbuf2)
                        # dedupe shard — bandingkan PATH PENUH (yang tersimpan
                        # di arena adalah path lengkap dengan prefix model_dir)
                        var full = model_dir + "/" + shard_name
                        var found = -1
                        for s in range(self.n_shards):
                            if self._shard_path_is(s, full):
                                found = s
                                break
                        if found == -1:
                            var sidx = self._add_shard_path(full)
                            self._scan_shard(full, sidx)
                            found = sidx
                        sbuf2.free()
                    pair = index_doc.next_sibling(pair)
            index_doc.free()
            self.ok = self.n_entries > 0
            return

        # 2. fallback file tunggal
        var single = model_dir + "/" + "model.safetensors"
        self._scan_shard(single, self._add_shard_path(single))
        self.ok = self.n_entries > 0

    fn find(self, name: String) -> Int:
        """Cari indeks entry berdasarkan nama tensor; -1 bila tidak ada."""
        var bytes = name.as_bytes()
        for i in range(self.n_entries):
            var e = self.entries[i]
            if len(bytes) == e.name_len:
                var same = True
                for j in range(e.name_len):
                    if self.names[e.name_off + j] != bytes[j]:
                        same = False
                        break
                if same:
                    return i
        return -1

    fn numel(self, entry_idx: Int) -> Int:
        var e = self.entries[entry_idx]
        return e.d0 * e.d1 * e.d2

    fn dim0(self, entry_idx: Int) -> Int:
        return self.entries[entry_idx].d0

    fn dim1(self, entry_idx: Int) -> Int:
        return self.entries[entry_idx].d1

    fn is_f32(self, entry_idx: Int) -> Bool:
        return self.entries[entry_idx].dtype == DT_F32

    # ---------------- pembacaan ----------------
    fn read_raw(self, entry_idx: Int, out_ptr: UnsafePointer[UInt8, MutAnyOrigin], nbytes: Int) raises -> Bool:
        """Salin mentah nbytes bobot terpaket (U8/U32) ke out_ptr. Seek absolut =
        8 + panjang header shard + start tensor (offset data section)."""
        var e = self.entries[entry_idx]
        var off = self.shard_offs[e.shard_idx]
        # arena path nul-terminated (lihat _add_shard_path); alias valid
        # selama indeks hidup — String hanya dipakai untuk open() di bawah.
        var path = String(unsafe_from_utf8_ptr=self.shard_paths + off)
        var handle = open(path, "r")
        var abs_off = UInt64(8 + self.shard_header_lens[e.shard_idx] + e.start)
        var _ = handle.seek(abs_off, 0)
        # Baca per-chunk 64 MiB: sekali read untuk tensor raksasa (1-2 GB)
        # berisiko gagal/terpotong pada beberapa runtime.
        var got_total = 0
        while got_total < nbytes:
            var want = nbytes - got_total
            if want > (1 << 26):
                want = 1 << 26
            var got = handle.read(Span[UInt8, MutAnyOrigin](ptr=out_ptr + got_total, length=want))
            if got <= 0:
                break
            got_total += got
        handle.close()
        return got_total == nbytes

    fn read_f32(self, entry_idx: Int, out_ptr: UnsafePointer[Float32, MutAnyOrigin], out_cap: Int) raises -> Bool:
        """Baca tensor dan konversi ke FP32 (F16/BF16 eksak, F32 salin)."""
        var e = self.entries[entry_idx]
        var elem = e.d0 * e.d1 * e.d2
        if elem != out_cap:
            return False
        var nbytes = e.nbytes
        var raw = alloc[UInt8](nbytes if nbytes > 0 else 1)
        var ok = self.read_raw(entry_idx, raw, nbytes)
        if not ok:
            raw.free()
            return False
        if e.dtype == DT_F32:
            for i in range(elem):
                var u = UInt32(raw[i * 4]) | (UInt32(raw[i * 4 + 1]) << 8) | (UInt32(raw[i * 4 + 2]) << 16) | (UInt32(raw[i * 4 + 3]) << 24)
                out_ptr[i] = bitcast[DType.float32](u)
        elif e.dtype == DT_F16:
            for i in range(elem):
                var u = UInt16(raw[i * 2]) | (UInt16(raw[i * 2 + 1]) << 8)
                out_ptr[i] = Float32(bitcast[DType.float16](u))
        elif e.dtype == DT_BF16:
            for i in range(elem):
                var u = UInt16(raw[i * 2]) | (UInt16(raw[i * 2 + 1]) << 8)
                out_ptr[i] = bitcast[DType.float32](UInt32(u) << 16)
        else:
            raw.free()
            return False
        raw.free()
        return True

# ---------------- fusion bobot terkuantisasi (fungsi bebas) ----------------
fn fuse_u8(
    a: UnsafePointer[UInt8, MutAnyOrigin], na: Int,
    b: UnsafePointer[UInt8, MutAnyOrigin], nb: Int
) -> UnsafePointer[UInt8, MutAnyOrigin]:
    """Gabung dua buffer bobot terpaket sepanjang N (byte concat)."""
    var fresh = alloc[UInt8](na + nb)
    for i in range(na):
        fresh[i] = a[i]
    for i in range(nb):
        fresh[na + i] = b[i]
    return fresh

fn fuse_f32(
    a: UnsafePointer[Float32, MutAnyOrigin], na: Int,
    b: UnsafePointer[Float32, MutAnyOrigin], nb: Int
) -> UnsafePointer[Float32, MutAnyOrigin]:
    """Gabung dua buffer skala FP32 sepanjang N (baris per grup g128)."""
    var fresh = alloc[Float32](na + nb)
    for i in range(na):
        fresh[i] = a[i]
    for i in range(nb):
        fresh[na + i] = b[i]
    return fresh

@fieldwise_init
struct LoadedQLinear(Copyable, Movable, ImplicitlyCopyable):
    """Hasil muat satu lapisan terkuantisasi: bobot packed + skala FP32."""
    var w: UnsafePointer[UInt8, MutAnyOrigin]
    var scales: UnsafePointer[Float32, MutAnyOrigin]
    var biases: UnsafePointer[Float32, MutAnyOrigin]
    var n_rows: Int
    var k_dim: Int
    var nbytes: Int
    var ok: Bool

fn load_qlinear(st: SafeTensorsIndex, w_name: String, s_name: String, b_name: String) raises -> LoadedQLinear:
    """Muat (weight packed, scales, biases) -> FP32. scales DIBAGI 2 di sini
    (s_eff = s_ckpt/2 untuk kernel (2q-1)*s); biases mentah untuk koreksi
    affine eksak di QwenLinear1Bit.forward (paritas mx.quantized_matmul)."""
    var r = LoadedQLinear(
        w=UnsafePointer[UInt8, MutAnyOrigin](), scales=UnsafePointer[Float32, MutAnyOrigin](),
        biases=UnsafePointer[Float32, MutAnyOrigin](),
        n_rows=0, k_dim=0, nbytes=0, ok=False
    )
    var ew = st.find(w_name)
    if ew == -1:
        ew = st.find("language_model." + w_name)
    var es = st.find(s_name)
    if es == -1:
        es = st.find("language_model." + s_name)
    var eb = st.find(b_name)
    if eb == -1:
        eb = st.find("language_model." + b_name)
    if ew == -1 or es == -1:
        return r
    var nb = st.entries[ew].nbytes
    var wb = alloc[UInt8](nb)
    if not st.read_raw(ew, wb, nb):
        wb.free()
        return r
    var nn = st.dim0(ew)
    var kk = nb * 8 // nn
    var snum = nn * (kk // 128)
    var sb = alloc[Float32](snum if snum > 0 else 1)
    if not st.read_f32(es, sb, snum):
        wb.free()
        sb.free()
        return r
    for i in range(snum):
        sb[i] *= Float32(0.5)
    var bb = alloc[Float32](snum if snum > 0 else 1)
    if eb != -1:
        var _ = st.read_f32(eb, bb, snum)
    return LoadedQLinear(w=wb, scales=sb, biases=bb, n_rows=nn, k_dim=kk, nbytes=nb, ok=True)
