# ===----------------------------------------------------------------------=== #
# Module: src/tokenizer.py
# Purpose: BPE Tokenizer Wrapper & Streaming Detokenizer untuk Qwen 3.5 / 3.6 / 3.8
# ===----------------------------------------------------------------------=== #

import os
import json
from typing import List, Optional, Generator

class QwenTokenizer:
    """
    Tokenizer wrapper untuk Qwen 3.5 / 3.6 / 3.8:
    Mendukung tokenizer Hugging Face (jika terpasang) dengan fallback native tokenizer.json.
    """

    # Cadangan bila config.json checkpoint tidak terbaca.
    # Terverifikasi thd config.json Bonsai-27B: eos_token_id=248046,
    # bos_token_id=248044; tokenizer_config.json: eos_token="<|im_end|>".
    # BUKAN 151643/151644/151645 — itu rentang Qwen2.5/Qwen3.0 (vocab 151936);
    # model ini generasi berikutnya dgn vocab 248320, ID-nya beda.
    EOS_TOKEN_ID = 248046       # <|im_end|>
    IM_START_TOKEN_ID = 248045  # <|im_start|>
    IM_END_TOKEN_ID = 248046    # <|im_end|>

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.hf_tokenizer = None
        self.eos_token_id = self.EOS_TOKEN_ID
        self.im_start_token_id = self.IM_START_TOKEN_ID
        self.im_end_token_id = self.IM_END_TOKEN_ID
        self._load_special_ids()
        self._load_tokenizer()
        self._resolve_special_ids_from_vocab()

    def _load_special_ids(self):
        """config.json checkpoint = otoritatif untuk eos id (bukan ingatan)."""
        path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(path):
            return
        try:
            cfg = json.load(open(path))
        except Exception:
            return
        cfg = cfg.get("text_config", cfg)
        if isinstance(cfg.get("eos_token_id"), int):
            self.eos_token_id = cfg["eos_token_id"]
            self.im_end_token_id = cfg["eos_token_id"]

    def _load_tokenizer(self):
        hf_err = None
        tok_err = None

        try:
            from transformers import AutoTokenizer
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
            print(">> [TOKENIZER] Berhasil memuat HuggingFace AutoTokenizer.")
            return
        except Exception as e:
            hf_err = e

        # Coba tokenizer.json native
        tok_json_path = os.path.join(self.model_dir, "tokenizer.json")
        if os.path.exists(tok_json_path):
            try:
                from tokenizers import Tokenizer
                self.hf_tokenizer = Tokenizer.from_file(tok_json_path)
                print(">> [TOKENIZER] Berhasil memuat native tokenizers.Tokenizer dari tokenizer.json.")
                return
            except Exception as e:
                tok_err = e

        raise RuntimeError(
            "Gagal memuat tokenizer dari " + self.model_dir
            + " (transformers: " + str(hf_err) + "; tokenizers: " + str(tok_err) + "). "
            "Fallback byte-level sudah dihapus: nilai byte 0..255 sebagai token id "
            "menghasilkan prompt yang salah secara senyap pada vocab 248320. "
            "Pasang `transformers` atau `tokenizers`, atau pakai model_dir yang benar."
        )

    def _resolve_special_ids_from_vocab(self):
        """Ambil nomor <|im_start|>/<|im_end|> dari vocab tokenizer sendiri.

        Alasan: config.json Bonsai-2 (paket 2-bit) menulis eos_token_id=248044
        yaitu <|endoftext|>, padahal <|im_end|> bernomor 248046 dan model itu
        benar-benar mengakhiri giliran dgn 248046. Karena itu im_end TIDAK
        boleh dipaksa sama dgn eos config. Pada Bonsai-27B (1-bit) keduanya
        sama-sama 248046, jadi hasilnya identik dgn perilaku lama.
        """
        if self.hf_tokenizer is None:
            return
        tok = self.hf_tokenizer
        for attr, text in (("im_start_token_id", "<|im_start|>"),
                           ("im_end_token_id", "<|im_end|>")):
            tid = None
            if hasattr(tok, "convert_tokens_to_ids"):
                tid = tok.convert_tokens_to_ids(text)
            elif hasattr(tok, "token_to_id"):
                tid = tok.token_to_id(text)
            if isinstance(tid, int) and tid >= 0:
                setattr(self, attr, tid)

    def encode(self, text: str) -> List[int]:
        """Mengubah string teks menjadi array token ID."""
        res = self.hf_tokenizer.encode(text)
        if hasattr(res, "ids"):
            return res.ids
        return res

    def decode(self, tokens: List[int]) -> str:
        """Mengubah array token ID menjadi string teks."""
        return self.hf_tokenizer.decode(tokens)

    def is_stop_token(self, token_id: int) -> bool:
        """Mengecek apakah token adalah stop/end-of-sequence token."""
        return token_id in (self.eos_token_id, self.im_end_token_id)


class StreamingDetokenizer:
    """
    Detokenizer bertahap (streaming) yang mencegah pemotongan karakter
    multi-byte UTF-8 saat token di-generate satu per satu.
    """

    def __init__(self, tokenizer: QwenTokenizer):
        self.tokenizer = tokenizer
        self.tokens: List[int] = []
        self.current_text = ""

    def add_token(self, token_id: int) -> str:
        """
        Menambahkan 1 token baru ke buffer dan mengembalikan teks tambahan (delta).
        """
        self.tokens.append(token_id)
        new_text = self.tokenizer.decode(self.tokens)
        
        # Hitung perbedaan string baru dengan string sebelumnya
        if len(new_text) > len(self.current_text):
            delta = new_text[len(self.current_text):]
            self.current_text = new_text
            return delta
        return ""

    def reset(self):
        self.tokens.clear()
        self.current_text = ""
