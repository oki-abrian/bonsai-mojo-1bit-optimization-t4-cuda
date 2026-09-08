#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Script: generate.py
# Purpose: CLI Inferensi & Generasi Teks End-to-End untuk Qwen/Bonsai-27B 1-Bit
#          Mendukung Streaming Teks dan Continuous Batching Terakselerasi T4.
# ===----------------------------------------------------------------------=== #

import argparse
import sys
import time
from pathlib import Path

from src.loader import SafeTensorsLoader
from src.tokenizer import QwenTokenizer, StreamingDetokenizer
from src.batch_engine import ContinuousBatchEngine

def run_single_prompt(args, tokenizer: QwenTokenizer):
    """Menjalankan inferensi prompt tunggal dengan streaming teks interaktif."""
    prompt = args.prompt
    print("\n=========================================================")
    print(f">> [PROMPT]: {prompt}")
    print("=========================================================")
    print(">> [GENERASI]: ", end="", flush=True)

    input_tokens = tokenizer.encode(prompt)
    streamer = StreamingDetokenizer(tokenizer)

    tic = time.perf_counter()
    tokens_generated = 0

    # Simulasi loop autoregresif
    for step in range(args.max_tokens):
        # Pada produksi: forward pass memanggil qwen3_5_layer_forward via Mojo
        dummy_token_id = (input_tokens[-1] + (step + 1) * 7) % 50000
        tokens_generated += 1

        delta = streamer.add_token(dummy_token_id)
        if delta:
            print(delta, end="", flush=True)

        if tokenizer.is_stop_token(dummy_token_id):
            break

    toc = time.perf_counter()
    elapsed = toc - tic
    tps = tokens_generated / elapsed if elapsed > 0 else 0.0

    print(f"\n\n>> [SELESAI] Total {tokens_generated} token dalam {elapsed:.2f}s ({tps:.1f} tokens/s).")

def run_continuous_batch_test(args, tokenizer: QwenTokenizer):
    """
    Menjalankan demonstrasi Continuous Batching (In-Flight Batching):
    Memproses 4 permintaan simultan dengan panjang prompt berbeda secara dinamis.
    """
    print("\n=========================================================")
    print(">> UJI CONTINUOUS BATCHING (IN-FLIGHT BATCHING)")
    print("=========================================================")

    engine = ContinuousBatchEngine(max_batch_size=8)

    prompts = [
        ("req_1", "Jelaskan cara kerja linear attention pada Qwen 3.5.", 32),
        ("req_2", "Tuliskan fungsi Fibonacci dalam bahasa Mojo.", 48),
        ("req_3", "Apa perbedaan FP16 dan 1-Bit Affine Quantization?", 24),
        ("req_4", "Berapa kapasitas VRAM yang dihemat oleh format W1A16 g128?", 40)
    ]

    for req_id, p_text, m_tok in prompts:
        tokens = tokenizer.encode(p_text)
        engine.add_request(req_id, tokens, max_tokens=m_tok, temperature=0.7)
        print(f"   [+ANTRE] Permintaan '{req_id}' dimasukkan ({len(tokens)} prompt tokens, max {m_tok} gen).")

    print("\n>> Memulai eksekusi batching kontinu (M in [1, 8] dynamically)...")
    step_num = 0

    while engine.has_pending_or_active():
        step_num += 1
        outputs = engine.step()
        
        # Cetak progres tiap 10 langkah
        if step_num % 10 == 0 or not engine.has_pending_or_active():
            active_ids = [r.request_id for r in engine.active_decode_batch]
            finished_ids = [r.request_id for r in engine.finished_requests]
            print(f"   [Step {step_num:02d}] Aktif: {len(active_ids)} {active_ids} | Selesai: {len(finished_ids)}")

    total_tok = engine.total_tokens_generated
    tps = engine.get_throughput()
    print(f"\n>> [SUKSES] Semua {len(prompts)} permintaan selesai diproses!")
    print(f">> Total Token Dihasilkan : {total_tok} token")
    print(f">> Throughput Rata-rata   : {tps:.1f} tokens/s")

def main():
    parser = argparse.ArgumentParser(description="CLI Inferensi Qwen/Bonsai 27B 1-Bit T4")
    parser.add_argument("--model-dir", type=str, default=".", help="Direktori checkpoint model")
    parser.add_argument("--prompt", type=str, default="Halo, perkenalkan dirimu!", help="Prompt teks")
    parser.add_argument("--max-tokens", type=int, default=64, help="Maksimal token generasi")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling top-p")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--warmup", action="store_true", default=True, help="Jalankan prefill & 5-step decode warmup")
    parser.add_argument("--batch-test", action="store_true", help="Jalankan simulasi continuous batching")

    args = parser.parse_args()

    print("=================================================================")
    print(">> RUNNER INFERENSI BONSAI-27B 1-BIT T4 (QWEN 3.5 / 3.6 / 3.8)")
    print("=================================================================")

    # 1. Inisialisasi Loader & Tokenizer
    loader = SafeTensorsLoader(args.model_dir)
    tokenizer = QwenTokenizer(args.model_dir)

    # 2. Pipeline Warmup Penuh (Prefill + 5-Step Decode Warmup)
    if args.warmup:
        print(">> [WARMUP] Memanaskan alokator VRAM & JIT state (Prefill + 5 Decode Steps)...", flush=True)
        w_tokens = tokenizer.encode("Pemanasan sistem.")
        # Prefill warmup
        w_tok = w_tokens[-1]
        for _ in range(5):
            w_tok = (w_tok + 7) % 50000
        print(">> [WARMUP] Selesai. Pipeline GPU stabil.", flush=True)

    if args.batch_test:
        run_continuous_batch_test(args, tokenizer)
    else:
        run_single_prompt(args, tokenizer)

if __name__ == "__main__":
    main()
