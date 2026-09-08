# ===----------------------------------------------------------------------=== #
# Module: src/models/qwen3_5/model.mojo
# Purpose: Pipeline model penuh, Token Embedding, dan LM Head
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer
from .norm import rms_norm

fn embed_tokens_step(
    hidden_states: UnsafePointer[Float32, MutAnyOrigin],
    token_id: Int,
    embedding_table: UnsafePointer[Float32, MutAnyOrigin],
    hidden_size: Int
):
    """Mengambil vektor embedding token dari tabel bobot embedding."""
    var offset = token_id * hidden_size
    for d in range(hidden_size):
        hidden_states[d] = embedding_table[offset + d]

fn qwen3_5_model_forward(
    hidden_states: UnsafePointer[Float32, MutAnyOrigin],
    final_norm_w: UnsafePointer[Float32, MutAnyOrigin],
    M: Int, D: Int, eps: Float32 = 1e-6
):
    """Normalisasi akhir RMSNorm sebelum kalkulasi logits token."""
    rms_norm(hidden_states, hidden_states, final_norm_w, M, D, eps)

fn lm_head_argmax_step(
    hidden_states: UnsafePointer[Float32, MutAnyOrigin],
    lm_head_weight: UnsafePointer[Float32, MutAnyOrigin],
    hidden_size: Int,
    vocab_size: Int
) -> Int:
    """Proyeksi LM Head dan penentuan token berikutnya via Greedy Argmax."""
    var best_token: Int = 0
    var best_logit: Float32 = -1e30

    for v in range(vocab_size):
        var logit: Float32 = 0.0
        var w_offset = v * hidden_size
        for d in range(hidden_size):
            logit += hidden_states[d] * lm_head_weight[w_offset + d]
        if logit > best_logit:
            best_logit = logit
            best_token = v

    return best_token
