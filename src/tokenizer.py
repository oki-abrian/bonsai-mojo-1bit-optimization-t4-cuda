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

    # Special Token ID Qwen
    EOS_TOKEN_ID = 151643     # <|endoftext|>
    IM_START_TOKEN_ID = 151644 # <|im_start|>
    IM_END_TOKEN_ID = 151645   # <|im_end|>

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.hf_tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            self.hf_tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
            print(">> [TOKENIZER] Berhasil memuat HuggingFace AutoTokenizer.")
            return
        except Exception:
            pass

        # Coba tokenizer.json native
        tok_json_path = os.path.join(self.model_dir, "tokenizer.json")
        if os.path.exists(tok_json_path):
            try:
                from tokenizers import Tokenizer
                self.hf_tokenizer = Tokenizer.from_file(tok_json_path)
                print(">> [TOKENIZER] Berhasil memuat native tokenizers.Tokenizer dari tokenizer.json.")
                return
            except Exception:
                pass

        print(">> [WARN] Library HuggingFace tokenizers tidak tersedia. Menggunakan fallback byte-level tokenizer.")

    def encode(self, text: str) -> List[int]:
        """Mengubah string teks menjadi array token ID."""
        if self.hf_tokenizer is not None:
            if hasattr(self.hf_tokenizer, "encode"):
                res = self.hf_tokenizer.encode(text)
                if hasattr(res, "ids"):
                    return res.ids
                return res
        # Fallback byte encoding
        return [b for b in text.encode("utf-8")]

    def decode(self, tokens: List[int]) -> str:
        """Mengubah array token ID menjadi string teks."""
        if self.hf_tokenizer is not None:
            if hasattr(self.hf_tokenizer, "decode"):
                return self.hf_tokenizer.decode(tokens)
        # Fallback byte decoding
        try:
            return bytes(tokens).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def is_stop_token(self, token_id: int) -> bool:
        """Mengecek apakah token adalah stop/end-of-sequence token."""
        return token_id in (self.EOS_TOKEN_ID, self.IM_END_TOKEN_ID)


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
