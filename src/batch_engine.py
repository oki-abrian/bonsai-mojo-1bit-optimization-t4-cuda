# ===----------------------------------------------------------------------=== #
# Module: src/batch_engine.py
# Purpose: Mesin Continuous Batching (In-Flight Batching) untuk Model Qwen 3.5 / 3.6 / 3.8
#          Diadaptasi langsung dari arsitektur BatchGenerator & Server MLX.
# ===----------------------------------------------------------------------=== #

import time
import math
import random
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass, field
from collections import deque

@dataclass
class SequenceRequest:
    """
    Representasi satu permintaan inferensi dalam sistem continuous batching.
    """
    request_id: str
    prompt_tokens: List[int]
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.05
    created_at: float = field(default_factory=time.time)

    # State Eksekusi
    generated_tokens: List[int] = field(default_factory=list)
    status: str = "PENDING" # "PENDING", "PREFILLING", "DECODING", "FINISHED"
    prefill_progress: int = 0
    finish_reason: Optional[str] = None

    # State Memori per Layer (GDN Matrix S + Attention KV Cache)
    gdn_states: List[Any] = field(default_factory=list)
    kv_caches: List[Any] = field(default_factory=list)


class ContinuousBatchEngine:
    """
    Mesin Continuous Batching (In-Flight Batching):
    Mengatur penjadwalan dinamis antara:
    1. Prefill Batch: Memproses prompt baru secara bertahap (chunking) via GEMM Prefill (M > 8).
    2. Active Decode Batch: Mengeksekusi generasi token aktif hingga M in [1, 8] via GEMV Decode.
    Request baru dapat disisipkan kapan saja tanpa harus menunggu request lama selesai!
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        prefill_chunk_size: int = 512,
        eos_token_id: int = 151643,
        im_end_token_id: int = 151645
    ):
        self.max_batch_size = max_batch_size
        self.prefill_chunk_size = prefill_chunk_size
        self.eos_token_id = eos_token_id
        self.im_end_token_id = im_end_token_id

        # Antrean Permintaan
        self.pending_queue: deque[SequenceRequest] = deque()
        self.active_decode_batch: List[SequenceRequest] = []
        self.finished_requests: List[SequenceRequest] = []

        # Statistik Kinerja
        self.total_tokens_generated = 0
        self.start_time = time.time()

    def add_request(
        self,
        request_id: str,
        prompt_tokens: List[int],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.05
    ) -> SequenceRequest:
        """Menambahkan permintaan inferensi baru ke dalam antrean penjadwal."""
        req = SequenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )
        self.pending_queue.append(req)
        return req

    def has_pending_or_active(self) -> bool:
        """Mengecek apakah masih ada permintaan yang sedang berjalan atau antre."""
        return len(self.pending_queue) > 0 or len(self.active_decode_batch) > 0

    def _sample_token(
        self,
        logits: List[float],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.05,
        past_token_ids: Optional[List[int]] = None
    ) -> int:
        """
        Sampler lengkap: Repetition Penalty, Greedy (Argmax), Temperature, Top-K, dan Nucleus (Top-P).
        Sesuai implementasi skrip inferensi optimal T4.
        """
        # 1. Repetition Penalty
        if repetition_penalty != 1.0 and past_token_ids:
            seen_tokens = set(past_token_ids)
            for token_id in seen_tokens:
                if 0 <= token_id < len(logits):
                    val = logits[token_id]
                    logits[token_id] = val * repetition_penalty if val < 0 else val / repetition_penalty

        # 2. Greedy Search
        if temperature <= 1e-4 or math.isclose(temperature, 0.0):
            return max(range(len(logits)), key=lambda i: logits[i])

        # 3. Temperature Scaling
        scaled = [l / temperature for l in logits]

        # 4. Top-K Filtering
        if top_k > 0 and top_k < len(scaled):
            k_indices = sorted(range(len(scaled)), key=lambda i: scaled[i], reverse=True)[:top_k]
            k_set = set(k_indices)
            scaled = [val if i in k_set else -1e9 for i, val in enumerate(scaled)]

        # 5. Top-P (Nucleus) Filter
        max_l = max(scaled)
        exp_l = [math.exp(l - max_l) for l in scaled]
        sum_exp = sum(exp_l)
        probs = [p / sum_exp for p in exp_l]

        sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        cumulative = 0.0
        filtered_indices = []
        filtered_probs = []

        for idx in sorted_indices:
            cumulative += probs[idx]
            filtered_indices.append(idx)
            filtered_probs.append(probs[idx])
            if cumulative >= top_p:
                break

        f_sum = sum(filtered_probs)
        norm_probs = [p / f_sum for p in filtered_probs]

        # Roulette selection
        r = random.random()
        acc = 0.0
        for idx, p in zip(filtered_indices, norm_probs):
            acc += p
            if r <= acc:
                return idx

        return filtered_indices[-1]

    def step(self) -> List[Tuple[str, int, bool]]:
        """
        Satu langkah iterasi continuous batching:
        1. Prefill Scheduling: Memasukkan request dari antrean ke active batch jika ada slot (kapasitas M <= 8).
        2. Decode Step: Menjalankan 1 langkah komputasi Decode GEMV secara simultan untuk seluruh request aktif.
        3. Sampling & Ejection: Mengecek stop token; jika selesai, keluarkan dari batch tanpa menghentikan request lain!
        
        Mengembalikan daftar tuple: [(request_id, token_id_baru, is_finished)]
        """
        # 1. Alokasikan slot kosong di active batch dari antrean pending
        while len(self.active_decode_batch) < self.max_batch_size and len(self.pending_queue) > 0:
            req = self.pending_queue.popleft()
            req.status = "DECODING"
            self.active_decode_batch.append(req)

        if not self.active_decode_batch:
            return []

        # 2. Ukuran batch dinamis saat ini (M in [1, 8])
        current_M = len(self.active_decode_batch)
        step_outputs = []
        still_active = []

        for req in self.active_decode_batch:
            pos = len(req.prompt_tokens) + len(req.generated_tokens)
            
            # Simulasi output logit layer terakhir model
            # (Pada integrasi penuh, ini memanggil model.forward(M=current_M))
            dummy_vocab_size = 152064
            # Logit sederhana deterministik
            sample_token_id = (req.prompt_tokens[-1] + pos * 13) % 50000

            req.generated_tokens.append(sample_token_id)
            self.total_tokens_generated += 1

            # 3. Evaluasi kondisi selesai (Stop Token atau Max Tokens)
            is_stop = sample_token_id in (self.eos_token_id, self.im_end_token_id)
            is_max = len(req.generated_tokens) >= req.max_tokens

            if is_stop or is_max:
                req.status = "FINISHED"
                req.finish_reason = "stop" if is_stop else "length"
                self.finished_requests.append(req)
                step_outputs.append((req.request_id, sample_token_id, True))
            else:
                still_active.append(req)
                step_outputs.append((req.request_id, sample_token_id, False))

        # Update batch aktif (request yang sudah selesai otomatis ter-eject!)
        self.active_decode_batch = still_active

        return step_outputs

    def get_throughput(self) -> float:
        """Menghitung total throughput token per detik (Tokens/s)."""
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.total_tokens_generated / elapsed
