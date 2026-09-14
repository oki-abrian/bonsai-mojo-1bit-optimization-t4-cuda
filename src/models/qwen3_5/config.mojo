# ===----------------------------------------------------------------------=== #
# Module: src/models/qwen3_5/config.mojo
# Purpose: Definisi konfigurasi arsitektur Qwen 3.5 / 3.6 / 3.8
# ===----------------------------------------------------------------------=== #

@fieldwise_init
struct QwenConfig(Copyable, Movable, ImplicitlyCopyable):
    """
    Parameter hiper-arsitektur model Qwen 3.5 / 3.6 / 3.8:
    Mendukung konfigurasi native untuk Qwen 27B, 4B, dan varian lainnya.

    Nilai pada komentar tiap field = nilai Bonsai-27B yang SEBENARNYA
    (`qwen_27b_default`, terverifikasi thd config.json checkpoint).
    """
    var hidden_size: Int               # Dimensi representasi token (27B: 5120)
    var intermediate_size: Int         # Dimensi proyeksi FFN SwiGLU (27B: 17408)
    var num_hidden_layers: Int         # Total lapisan transformer (27B: 64)
    var num_attention_heads: Int       # Head query full attention (27B: 24)
    var num_key_value_heads: Int       # Head KV pada full attention GQA (27B: 4)
    var head_dim: Int                  # Dimensi fitur per-head (27B: 256)
    var vocab_size: Int                # Ukuran kosakata tokenizer (27B: 248320)
    var rms_norm_eps: Float32          # Epsilon stabilitas RMSNorm (27B: 1e-6)
    var full_attention_interval: Int   # Rasio hybrid 3:1 (4 -> 3 GDN : 1 Full Attn)

    # Parameter Rotary Positional Embedding (RoPE)
    var rope_theta: Float32            # Basis frekuensi geometrik RoPE (27B: 1e7)
    var partial_rotary_factor: Float32 # Fraksi dimensi head yang dirotasi (27B: 0.25)
    var rotary_dim: Int                # Dimensi nyata yang dirotasi (27B: 64)

    # Parameter Linear Attention Gated DeltaNet (GDN)
    var gdn_num_v_heads: Int           # Jumlah head Value pada GDN (27B: 48)
    var gdn_num_k_heads: Int           # Jumlah head Key pada GDN (27B: 16)
    var gdn_head_k_dim: Int            # Dimensi head Key GDN (27B: 128)
    var gdn_head_v_dim: Int            # Dimensi head Value GDN (27B: 128)
    var gdn_conv_kernel: Int           # Ukuran window causal convolution (27B: 4)
    var gdn_conv_dim: Int              # Dimensi gabungan QKV konvolusi (27B: 10240)

    @staticmethod
    fn qwen_27b_default() -> QwenConfig:
        """Konfigurasi default Qwen 27B (Qwen 3.5 / Qwen 3.6 / Qwen 3.8)."""
        var h_dim = 256
        var rot_factor = Float32(0.25)
        var rot_dim = Int(Float32(h_dim) * rot_factor) # 64
        # qwen3_5 (config.json checkpoint asli): 48 v-head x 128, 16 k-head x 128
        var conv_dim = 2 * (16 * 128) + (48 * 128)     # 10240
        return QwenConfig(
            hidden_size=5120,
            intermediate_size=17408,
            num_hidden_layers=64,
            num_attention_heads=24,
            num_key_value_heads=4,
            head_dim=h_dim,
            vocab_size=248320,
            rms_norm_eps=1e-6,
            full_attention_interval=4,
            rope_theta=Float32(10000000.0),
            partial_rotary_factor=rot_factor,
            rotary_dim=rot_dim,
            gdn_num_v_heads=48,
            gdn_num_k_heads=16,
            gdn_head_k_dim=128,
            gdn_head_v_dim=128,
            gdn_conv_kernel=4,
            gdn_conv_dim=conv_dim
        )
