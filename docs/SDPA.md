# PyTorch Scaled Dot Product Attention

BareLLM uses `torch.nn.functional.scaled_dot_product_attention` (SDPA) as its
portable attention backend. The model prepares Q, K, and V tensors and applies
RoPE; an attention backend then owns the cache update, KV-head expansion, mask
selection, and attention weighting/value aggregation.

The reference API is documented by PyTorch:

<https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>

## Tensor contract

BareLLM passes tensors with this layout:

```text
Q: [batch, query_heads, query_length, head_dim]
K: [batch, query_heads, key_length, head_dim]
V: [batch, query_heads, key_length, head_dim]
```

The attention result has shape:

```text
[batch, query_heads, query_length, head_dim]
```

For MHA, the query and KV head counts are equal. For GQA and MQA, BareLLM
projects fewer KV heads and repeats them across query-head groups before SDPA:

```text
MHA: Q=4, K/V=4
GQA: Q=4, K/V=2  -> each KV head serves 2 query heads
MQA: Q=4, K/V=1  -> the KV head serves all 4 query heads
```

This means the KV cache stores `num_kv_heads`, not `num_heads`.

## Causal attention

For a single-request multi-token prefill, the per-request cache returns no
explicit mask, so the SDPA backend enables causal mode:

```python
F.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=None,
    is_causal=True,
)
```

The causal rule is:

```text
query position i may attend to key positions 0 .. i
```

This prevents a token from reading future tokens in the same sequence.

For one-token cached decode, causal mode is deliberately disabled:

```python
F.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=None,
    is_causal=False,
)
```

The key sequence contains past tokens plus the current token, and the newest
query is allowed to attend to all of them. PyTorch's non-square causal mask is
upper-left aligned, so `is_causal=True` would incorrectly hide the current
token in this case.

## Attention masks

SDPA supports either an explicit `attn_mask` or `is_causal=True`. BareLLM does
not rely on both at the same time. When `attn_mask` is present, the model sets
`is_causal=False`.

For a boolean SDPA mask:

```text
True  = this key participates in attention
False = this key is masked out
```

This is the opposite of the convention used by some padding-mask APIs, where
`True` means “masked”.

BareLLM currently uses an explicit mask for unequal-length one-token batched
decode. For example, if two requests have cache lengths 2 and 4, then after
one new token their key lengths are 3 and 5:

```text
request A: [True, True, True, False, False]
request B: [True, True, True, True,  True]
```

The padded keys contain zeros, but the mask is still required because zeros
are valid numeric values and must not participate in the softmax.

## Prefill versus decode

BareLLM currently has two supported attention shapes:

### Prefill

Prefill processes multiple prompt tokens at once for one request:

```text
Q: [1, heads, prompt_length, head_dim]
K: [1, heads, prompt_length, head_dim]
```

The cache receives the prompt K/V, but its single-request view returns no
explicit attention mask. SDPA's causal mode provides the triangular attention
pattern.

### Decode

Decode processes one newly selected token per request:

```text
Q: [batch, heads, 1, head_dim]
K: [batch, heads, cached_length + 1, head_dim]
```

Each query is the newest position, so it may attend to the complete cached
history. In a batch with unequal histories, `BatchKVCache` pads K/V to a common
length and supplies the explicit valid-key mask.

The current engine intentionally supports one-token batched decode. It does
not yet claim support for multi-token cached batched prefill.

## Backend selection

Attention modules accept an `attention_backend` setting when the model is
constructed:

```text
sdpa    always use dense K/V plus PyTorch SDPA
auto    currently resolves to sdpa; future capability-gated selection
triton  reserved for the CUDA direct-paged backend and currently errors
```

`sdpa` is the default. Backend selection is injected through model/runtime
construction; there is no mutable global backend. Triton is optional and is
not imported by the CPU/MPS path.

## KV-cache path

The attention module does not know whether storage is contiguous or paged. The
selected backend receives the model-facing cache interface described in
[CACHE.md](CACHE.md) and owns the append-plus-attend operation:

```text
attention projections + RoPE
  -> backend.attend(q, new_k, new_v, layer_cache, group_size)
  -> LayerKV.append(new_k, new_v)       # SDPA backend
  -> complete logical K/V history
  -> KV-head expansion
  -> SDPA
```

`ContiguousKVCache` stores dense history directly. `PagedKVCache` writes each
token into a physical block and gathers the logical history into dense K/V
tensors before calling SDPA:

```text
logical position
  -> logical block index
  -> request block table
  -> physical block
  -> gathered dense K/V
  -> SDPA
```

Direct attention over physical pages is a future optimization. The current
dense-gather path keeps the attention implementation easy to compare against
the contiguous reference cache and is the correctness oracle for future CUDA
backends.

The cache also exposes an `append_only()` seam and checked post-append paged
metadata. The SDPA backend does not need that metadata: it intentionally uses
the legacy gather-producing `append()`. A future direct backend will use
`append_only()` and read the physical pages through the metadata, so the token
is written exactly once and no dense gather is performed unnecessarily.

## CPU and MPS behavior

CPU and Apple MPS use the same PyTorch SDPA reference path. Upstream Triton is
not required, and no Triton import occurs when using `sdpa` or `auto`. MPS
benchmarks must synchronize before reading timings; unsupported-operation CPU
fallback should not be enabled when claiming MPS performance because it can
hide device transfers.

## Uncached reference execution

When `kv_cache=None`, attention receives the complete sequence on every model
call. For a prompt of length `T`, prefill uses:

```text
Q/K/V: [1, heads, T, head_dim]
```

After sampling one token, the next decode call recomputes the sequence of
length `T + 1`, then `T + 2`, and so on. Because each call contains the whole
sequence, SDPA's causal mode is sufficient and no padding mask or cache mask is
needed. This is the slow reference path used to verify cached decoding.

## Grouped-query attention

BareLLM currently repeats KV heads explicitly:

```python
k = k.repeat_interleave(group_size, dim=1)
v = v.repeat_interleave(group_size, dim=1)
```

After repetition, SDPA sees the same number of query, key, and value heads.
This keeps the cache interface simple and makes MHA, GQA, and MQA share the
same attention kernel call.

PyTorch also exposes an `enable_gqa` option in SDPA. BareLLM does not use that
option because explicit repetition makes the head layout visible and works
with the current model-side interface.

## Backend and dropout notes

SDPA may select different internal implementations depending on device and
input properties. BareLLM relies on PyTorch's dispatch rather than selecting a
specific kernel.

The model is an inference model and does not pass a nonzero dropout
probability. SDPA applies dropout based on its `dropout_p` argument, so a
training/evaluation mode switch alone is not sufficient to disable dropout if
that argument is nonzero.

BareLLM configures CPU, CUDA, and MPS execution paths. Numerical results may
differ slightly between devices or SDPA kernels. The current equivalence tests
run with explicit tolerances rather than exact equality. The SDPA path is the
reference against which a future Triton implementation must be compared.

## Verification in this repository

The SDPA contract is exercised by:

- `tests/test_attention.py` for causal behavior, shapes, and head grouping;
- `tests/test_cache_equivalence.py` for contiguous cached MHA/GQA/MQA decode;
- `tests/test_qwen3.py` for multi-step paged cached MHA/GQA/MQA decode;
- `tests/test_batched_kv_cache.py` for real paged batched MHA/GQA/MQA decode;
- `tests/test_contiguous_kv_cache.py` and `tests/test_paged_kv_cache.py` for
  storage behavior independent of model math.
- `tests/test_attention_backend.py` for backend selection and CPU/MPS device
  behavior.
- `examples/benchmark_paged_cache.py` for separate page-gather and
  gather+SDPA measurements.

The intended reference rule is simple:

```text
full uncached logits == cached logits
full independent logits == batched logits
```

within the tolerance required by the active dtype and device.
