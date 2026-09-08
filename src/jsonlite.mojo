# ===----------------------------------------------------------------------=== #
# Module: jsonlite.mojo
# Purpose: Parser JSON minimal rekursif-descent berbasis arena memori.
#          Cukup untuk: header safetensors, config.json, dan tokenizer.json
#          (object/array bersarang, string ber-escape, angka, bool, null).
#          Tanpa dependensi di luar std; error ditandai lewat return value.
# Catatan port: gaya import tanpa prefix `std.` mengikuti konvensi repo;
#          docs v1.0 memakai prefix `std.` — akan terverifikasi di kompilasi
#          pertama (kedua bentuk diharapkan valid).
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc

# Tipe node JSON
alias JSON_OBJ: UInt8 = 0
alias JSON_ARR: UInt8 = 1
alias JSON_STR: UInt8 = 2
alias JSON_NUM: UInt8 = 3
alias JSON_BOOL: UInt8 = 4
alias JSON_NULL: UInt8 = 5
alias JSON_PAIR: UInt8 = 6 # elemen object: key (str_off/len) + value (first_child)

@fieldwise_init
struct JsonNode(Copyable, Movable, ImplicitlyCopyable):
    var kind: UInt8
    var first_child: Int  # untuk obj/arr/pair; -1 jika tidak ada
    var next_sibling: Int # -1 jika terakhir
    var str_off: Int      # offset ke buffer string (key untuk PAIR)
    var str_len: Int
    var num: Float64

struct JsonDoc:
    """
    Dokumen JSON hasil parse + arena node/buffer string miliknya.
    Pemanggil wajib memanggil free() setelah selesai.
    """
    var nodes: UnsafePointer[JsonNode, MutAnyOrigin]
    var n_nodes: Int
    var cap_nodes: Int
    var sbuf: UnsafePointer[UInt8, MutAnyOrigin]
    var n_str: Int
    var cap_str: Int
    var src: UnsafePointer[UInt8, MutAnyOrigin]
    var src_len: Int
    var pos: Int
    var depth: Int

    fn __init__(out self):
        self.nodes = UnsafePointer[JsonNode, MutAnyOrigin]()
        self.n_nodes = 0
        self.cap_nodes = 0
        self.sbuf = UnsafePointer[UInt8, MutAnyOrigin]()
        self.n_str = 0
        self.cap_str = 0
        self.src = UnsafePointer[UInt8, MutAnyOrigin]()
        self.src_len = 0
        self.pos = 0
        self.depth = 0

    fn free(mut self):
        if self.n_nodes > 0:
            self.nodes.free()
            self.nodes = UnsafePointer[JsonNode, MutAnyOrigin]()
            self.n_nodes = 0
            self.cap_nodes = 0
        if self.cap_str > 0:
            self.sbuf.free()
            self.sbuf = UnsafePointer[UInt8, MutAnyOrigin]()
            self.n_str = 0
            self.cap_str = 0

    # ---------------- arena ----------------
    fn _grow_nodes(mut self):
        var new_cap = 4096 if self.cap_nodes == 0 else self.cap_nodes * 2
        var fresh = alloc[JsonNode](new_cap)
        for i in range(self.n_nodes):
            fresh[i] = self.nodes[i]
        if self.cap_nodes > 0:
            self.nodes.free()
        self.nodes = fresh
        self.cap_nodes = new_cap

    fn _append_node(mut self, kind: UInt8) -> Int:
        if self.n_nodes == self.cap_nodes:
            self._grow_nodes()
        var idx = self.n_nodes
        self.nodes[idx] = JsonNode(
            kind=kind, first_child=-1, next_sibling=-1, str_off=0, str_len=0, num=0.0
        )
        self.n_nodes += 1
        return idx

    fn _grow_str(mut self, extra: Int):
        var needed = self.n_str + extra
        if needed <= self.cap_str:
            return
        var new_cap = 65536 if self.cap_str == 0 else self.cap_str * 2
        while new_cap < needed:
            new_cap *= 2
        var fresh = alloc[UInt8](new_cap)
        for i in range(self.n_str):
            fresh[i] = self.sbuf[i]
        if self.cap_str > 0:
            self.sbuf.free()
        self.sbuf = fresh
        self.cap_str = new_cap

    fn _append_str_byte(mut self, b: UInt8):
        self._grow_str(1)
        self.sbuf[self.n_str] = b
        self.n_str += 1

    # ---------------- parser ----------------
    fn parse_bytes(mut self, data: UnsafePointer[UInt8, MutAnyOrigin], length: Int) -> Bool:
        """Parse buffer JSON; True bila sukses. Root ada di self.root()."""
        self.src = data
        self.src_len = length
        self.pos = 0
        self.depth = 0
        self._skip_ws()
        var r = self._parse_value()
        if r == -1:
            return False
        self._skip_ws()
        return self.pos == self.src_len

    fn root(self) -> Int:
        """Indeks node root: node pertama yang di-parse."""
        return 0

    fn first_child(self, node: Int) -> Int:
        """Anak pertama node (obj/arr/pair); -1 jika tidak ada."""
        return self.nodes[node].first_child

    fn next_sibling(self, node: Int) -> Int:
        """Saudara berikutnya; -1 jika terakhir."""
        return self.nodes[node].next_sibling

    fn _skip_ws(mut self):
        while self.pos < self.src_len:
            var b = self.src[self.pos]
            if b == 32 or b == 9 or b == 10 or b == 13:
                self.pos += 1
            else:
                break

    fn _peek(self) -> UInt8:
        return self.src[self.pos] if self.pos < self.src_len else UInt8(0)

    fn _parse_value(mut self) -> Int:
        """Parse satu nilai JSON, kembalikan indeks node (-1 bila gagal)."""
        if self.depth > 64:
            return -1
        self._skip_ws()
        var b = self._peek()
        if b == UInt8(ord("{")):
            return self._parse_object()
        if b == UInt8(ord("[")):
            return self._parse_array()
        if b == UInt8(ord("\"")):
            return self._parse_string_value()
        if b == UInt8(ord("t")):
            return self._parse_lit("true", 1)
        if b == UInt8(ord("f")):
            return self._parse_lit("false", 0)
        if b == UInt8(ord("n")):
            return self._parse_lit("null", -1)
        return self._parse_number()

    fn _parse_lit(mut self, word: String, as_bool_or_neg: Int) -> Int:
        # Cocokkan kata literal (true/false/null) pada posisi sekarang.
        var wb = word.as_bytes()
        var n = len(wb)
        if self.pos + n > self.src_len:
            return -1
        for i in range(n):
            if self.src[self.pos + i] != wb[i]:
                return -1
        self.pos += n
        var node = self._append_node(JSON_BOOL if as_bool_or_neg >= 0 else JSON_NULL)
        self.nodes[node].num = Float64(as_bool_or_neg) if as_bool_or_neg >= 0 else 0.0
        return node

    fn _parse_object(mut self) -> Int:
        self.pos += 1 # lewati '{'
        self.depth += 1
        var obj = self._append_node(JSON_OBJ)
        self._skip_ws()
        if self._peek() == UInt8(ord("}")):
            self.pos += 1
            self.depth -= 1
            return obj
        var last = -1
        while True:
            self._skip_ws()
            if self._peek() != UInt8(ord("\"")):
                self.depth -= 1
                return -1
            # key: parse string mentah lalu salin sebagai PAIR
            var key_node = self._parse_string_value()
            if key_node == -1:
                self.depth -= 1
                return -1
            self._skip_ws()
            if self._peek() != UInt8(ord(":")):
                self.depth -= 1
                return -1
            self.pos += 1
            var val = self._parse_value()
            if val == -1:
                self.depth -= 1
                return -1
            # jadikan node string key sebagai PAIR yang membungkus value
            self.nodes[key_node].kind = JSON_PAIR
            self.nodes[key_node].first_child = val
            if last == -1:
                self.nodes[obj].first_child = key_node
            else:
                self.nodes[last].next_sibling = key_node
            last = key_node
            self._skip_ws()
            var c = self._peek()
            if c == UInt8(ord(",")):
                self.pos += 1
                continue
            if c == UInt8(ord("}")):
                self.pos += 1
                self.depth -= 1
                return obj
            self.depth -= 1
            return -1

    fn _parse_array(mut self) -> Int:
        self.pos += 1 # lewati '['
        self.depth += 1
        var arr = self._append_node(JSON_ARR)
        self._skip_ws()
        if self._peek() == UInt8(ord("]")):
            self.pos += 1
            self.depth -= 1
            return arr
        var last = -1
        while True:
            var val = self._parse_value()
            if val == -1:
                self.depth -= 1
                return -1
            if last == -1:
                self.nodes[arr].first_child = val
            else:
                self.nodes[last].next_sibling = val
            last = val
            self._skip_ws()
            var c = self._peek()
            if c == UInt8(ord(",")):
                self.pos += 1
                continue
            if c == UInt8(ord("]")):
                self.pos += 1
                self.depth -= 1
                return arr
            self.depth -= 1
            return -1

    fn _parse_string_value(mut self) -> Int:
        """Parse string JSON (dengan escape) ke node JSON_STR."""
        if self._peek() != UInt8(ord("\"")):
            return -1
        self.pos += 1
        var node = self._append_node(JSON_STR)
        self.nodes[node].str_off = self.n_str
        while True:
            if self.pos >= self.src_len:
                return -1
            var b = self.src[self.pos]
            self.pos += 1
            if b == UInt8(ord("\"")):
                self.nodes[node].str_len = self.n_str - self.nodes[node].str_off
                return node
            if b != UInt8(ord("\\")):
                self._append_str_byte(b)
                continue
            # escape
            if self.pos >= self.src_len:
                return -1
            var esc = self.src[self.pos]
            self.pos += 1
            if esc == UInt8(ord("\"")):
                self._append_str_byte(UInt8(ord("\"")))
            elif esc == UInt8(ord("\\")):
                self._append_str_byte(UInt8(ord("\\")))
            elif esc == UInt8(ord("/")):
                self._append_str_byte(UInt8(ord("/")))
            elif esc == UInt8(ord("b")):
                self._append_str_byte(8)
            elif esc == UInt8(ord("f")):
                self._append_str_byte(12)
            elif esc == UInt8(ord("n")):
                self._append_str_byte(10)
            elif esc == UInt8(ord("r")):
                self._append_str_byte(13)
            elif esc == UInt8(ord("t")):
                self._append_str_byte(9)
            elif esc == UInt8(ord("u")):
                if self.pos + 4 > self.src_len:
                    return -1
                var cp = self._hex4(self.pos)
                if cp == -1:
                    return -1
                self.pos += 4
                # surrogate pair
                if cp >= 0xD800 and cp <= 0xDBFF:
                    if self.pos + 6 <= self.src_len and self.src[self.pos] == UInt8(ord("\\")) and self.src[self.pos + 1] == UInt8(ord("u")):
                        var lo = self._hex4(self.pos + 2)
                        if lo >= 0xDC00 and lo <= 0xDFFF:
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                            self.pos += 6
                self._append_utf8(cp)
            else:
                return -1

    fn _hex4(self, at: Int) -> Int:
        var v = 0
        for i in range(4):
            if at + i >= self.src_len:
                return -1
            var b = self.src[at + i]
            var d = 0
            if b >= UInt8(ord("0")) and b <= UInt8(ord("9")):
                d = Int(b) - Int(ord("0"))
            elif b >= UInt8(ord("a")) and b <= UInt8(ord("f")):
                d = Int(b) - Int(ord("a")) + 10
            elif b >= UInt8(ord("A")) and b <= UInt8(ord("F")):
                d = Int(b) - Int(ord("A")) + 10
            else:
                return -1
            v = v * 16 + d
        return v

    fn _append_utf8(mut self, cp: Int):
        # encode codepoint UTF-8 (maksimal 4 byte)
        if cp < 0x80:
            self._append_str_byte(UInt8(cp))
        elif cp < 0x800:
            self._append_str_byte(UInt8(0xC0 | (cp >> 6)))
            self._append_str_byte(UInt8(0x80 | (cp & 0x3F)))
        elif cp < 0x10000:
            self._append_str_byte(UInt8(0xE0 | (cp >> 12)))
            self._append_str_byte(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            self._append_str_byte(UInt8(0x80 | (cp & 0x3F)))
        else:
            self._append_str_byte(UInt8(0xF0 | (cp >> 18)))
            self._append_str_byte(UInt8(0x80 | ((cp >> 12) & 0x3F)))
            self._append_str_byte(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            self._append_str_byte(UInt8(0x80 | (cp & 0x3F)))

    fn _parse_number(mut self) -> Int:
        """Parse angka JSON (int/float/eksponen) dalam satu lintasan digit."""
        var start = self.pos
        var neg = False
        if self.src[self.pos] == UInt8(ord("-")):
            neg = True
            self.pos += 1
        elif self.src[self.pos] == UInt8(ord("+")):
            self.pos += 1
        var value: Float64 = 0.0
        var seen_dot = False
        var frac_scale = 0.1
        var digits = 0
        while self.pos < self.src_len:
            var b = self.src[self.pos]
            if b >= UInt8(ord("0")) and b <= UInt8(ord("9")):
                if seen_dot:
                    value += Float64(Int(b) - Int(ord("0"))) * frac_scale
                    frac_scale *= 0.1
                else:
                    value = value * 10.0 + Float64(Int(b) - Int(ord("0")))
                digits += 1
                self.pos += 1
            elif b == UInt8(ord(".")) and not seen_dot:
                seen_dot = True
                self.pos += 1
            else:
                break
        if digits == 0:
            self.pos = start
            return -1
        # eksponen opsional: e/E [+/-] digit
        if self.pos < self.src_len and (self.src[self.pos] == UInt8(ord("e")) or self.src[self.pos] == UInt8(ord("E"))):
            self.pos += 1
            var eneg = False
            if self.pos < self.src_len and self.src[self.pos] == UInt8(ord("-")):
                eneg = True
                self.pos += 1
            elif self.pos < self.src_len and self.src[self.pos] == UInt8(ord("+")):
                self.pos += 1
            var e: Int = 0
            var e_digits = 0
            while self.pos < self.src_len and self.src[self.pos] >= UInt8(ord("0")) and self.src[self.pos] <= UInt8(ord("9")):
                e = e * 10 + Int(self.src[self.pos]) - Int(ord("0"))
                e_digits += 1
                self.pos += 1
            if e_digits > 0:
                var m = 1.0
                for _ in range(e):
                    m *= 10.0
                if eneg:
                    value /= m
                else:
                    value *= m
        if neg:
            value = -value

        var node = self._append_node(JSON_NUM)
        self.nodes[node].num = value
        return node

    # ---------------- aksesor ----------------
    fn kind(self, node: Int) -> UInt8:
        return self.nodes[node].kind

    fn arr_len(self, node: Int) -> Int:
        """Jumlah anak array; untuk object mengembalikan jumlah pasangan."""
        var count = 0
        var c = self.nodes[node].first_child
        while c != -1:
            count += 1
            c = self.nodes[c].next_sibling
        return count

    fn child_at(self, node: Int, slot: Int) -> Int:
        """Anak ke-slot (array) atau value pasangan ke-slot (object)."""
        var c = self.nodes[node].first_child
        var i = 0
        while c != -1:
            if i == slot:
                return c
            i += 1
            c = self.nodes[c].next_sibling
        return -1

    fn obj_get(self, obj: Int, key: String) -> Int:
        """Cari pasangan ber-key tertentu pada object; kembalikan node value."""
        var c = self.nodes[obj].first_child
        while c != -1:
            if self._key_equals(c, key):
                return self.nodes[c].first_child
            c = self.nodes[c].next_sibling
        return -1

    fn _key_equals(self, pair: Int, key: String) -> Bool:
        var off = self.nodes[pair].str_off
        var klen = self.nodes[pair].str_len
        var bytes = key.as_bytes()
        if len(bytes) != klen:
            return False
        for i in range(klen):
            if self.sbuf[off + i] != bytes[i]:
                return False
        return True

    fn str_len_of(self, node: Int) -> Int:
        return self.nodes[node].str_len

    fn str_copy(self, node: Int, out_ptr: UnsafePointer[UInt8, MutAnyOrigin], out_cap: Int) -> Int:
        """Salin isi string node ke out_ptr; kembalikan panjang (bisa terpotong)."""
        var n = self.nodes[node].str_len
        var copy = n if n < out_cap else out_cap
        for i in range(copy):
            out_ptr[i] = self.sbuf[self.nodes[node].str_off + i]
        return n

    fn as_int(self, node: Int) -> Int:
        return Int(self.nodes[node].num)

    fn as_f64(self, node: Int) -> Float64:
        return self.nodes[node].num
