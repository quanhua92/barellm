# KV-Cache Design

BareLLM separates the cache interface seen by the model from the storage and
lifecycle machinery owned by the engine. Attention only asks for a layer cache;
it does not allocate blocks or know whether K/V tensors are stored contiguously
or in pages.

This document describes the current implementation. Direct paged attention,
prefix sharing, eviction, and quantized KV storage remain future work.

## Why cache K and V?

During autoregressive decoding, every new token attends to all earlier tokens.
The K and V projections for those earlier tokens do not change, so recomputing
them on every step is wasteful. The cache stores them once and appends the K/V
for each newly processed token.

For each transformer layer, the model-facing shape is:

```text
[batch, num_kv_heads, sequence_length, head_dim]
```

The cache stores `num_kv_heads`, not the number of query heads. This matters for
GQA and MQA, where multiple query heads share one KV head.

## Architectural boundary

The cache is split across model and engine modules:

```text
models/cache.py
  LayerKV protocol
  KVCache protocol
          |
          | structural interface
          v
engine/contiguous_kv_cache.py   reference dense storage
engine/paged_kv_cache.py        physical paged storage + request views
engine/batched_kv_cache.py      temporary adapter over request caches
engine/kv_cache_manager.py      request allocation and lifetime
engine/block_pool.py            physical block-ID capacity
```

This direction is deliberate: engine implementations depend on the model-side
protocols, while model code does not depend on an engine storage class.

## Model-facing protocols

[`models/cache.py`](../src/barellm/models/cache.py) defines two structural
protocols.

### `KVCache`

```python
class KVCache(Protocol):
    def layer(self, layer_idx: int) -> LayerKV: ...
```

A model receives one `KVCache` and requests the view for its current transformer
layer. The model owns `layer_idx`; callers do not pass one cache object per
layer.

### `LayerKV`

```python
class LayerKV(Protocol):
    @property
    def seq_len(self) -> int: ...

    def append(self, key, value) -> tuple[Tensor, Tensor]: ...
    def read(self) -> tuple[Tensor, Tensor]: ...
    def attention_mask(self, q_len: int) -> Tensor | None: ...
```

The methods mean:

- `seq_len` is the logical number of cached tokens for that layer;
- `append` mutates the cache and returns the complete logical K/V tensors that
  attention should use for the current call;
- `read` returns the current logical K/V history without appending;
- `attention_mask` returns storage/batch-specific key validity information, or
  `None` when no explicit mask is required.

Returning tensors from `append` is important. Attention consumes the tensors
returned for this operation instead of appending and then assuming that the
storage representation can be read directly. A future compressed or quantized
cache can therefore choose how to produce the tensors needed by attention.

Protocols use structural typing. Cache classes satisfy the interface without
inheriting from a shared base class.

## Attention interaction

Both MHA and GQA use the same cache sequence in
[`models/attention.py`](../src/barellm/models/attention.py):

```text
project new Q, K, V
  -> apply QK norm when configured
  -> apply RoPE to Q and new K
  -> layer = kv_cache.layer(layer_idx)
  -> full_k, full_v = layer.append(new_k, new_v)
  -> mask = layer.attention_mask(q_len)
  -> SDPA(Q, full_k, full_v, mask)
```

Only K and V are cached. Q belongs to the current model invocation. RoPE is
applied before K enters the cache, so cached keys already contain their position
rotation.

For GQA and MQA, storage returns `num_kv_heads`; attention repeats those heads
to the query-head count after cache append and before SDPA.

## Contiguous reference cache

[`engine/contiguous_kv_cache.py`](../src/barellm/engine/contiguous_kv_cache.py)
contains the simplest implementation:

- `ContiguousKVCache` owns one `ContiguousLayerKV` per transformer layer;
- the first append stores K/V tensors directly;
- later appends concatenate along the sequence dimension;
- `read` returns the dense tensors;
- `attention_mask` returns `None` because one request has no padding.

Conceptually:

```text
before: [B, Hkv, T_old, D]
new:    [B, Hkv, T_new, D]
after:  [B, Hkv, T_old + T_new, D]
```

This backend is intentionally simple and inefficient for long decode sequences:
every `torch.cat` allocates a new tensor and copies the existing history. Its
role is to provide a clear reference path for correctness tests.

## Paged storage

[`engine/paged_kv_cache.py`](../src/barellm/engine/paged_kv_cache.py) preallocates
separate K and V tensors with layout:

```text
[num_layers, max_blocks, num_kv_heads, block_size, head_dim]
```

The dimensions mean:

```text
num_layers       transformer layer
max_blocks       physical blocks shared by all requests
num_kv_heads     stored K/V heads
block_size       token slots in one physical block
head_dim         values in one head
```

Paged storage currently supports batch size one per request view. Multi-request
batching is built by composing multiple request views with `BatchKVCache`.

### Request page state

For each registered request, `PagedKVCache` stores:

```text
blocks:          ordered physical blocks for the request
layer_seq_lens:  logical sequence length for every layer
```

Layer lengths are separate because every transformer layer appends during its
own forward pass. They normally advance together, but storage should not infer
one layer's state from another.

When a known request receives more blocks, `register_request` replaces its block
list while preserving `layer_seq_lens`. A new request starts with zero lengths.

### Logical-to-physical mapping

For logical token position `p`:

```text
logical_block = p // block_size
block_offset  = p % block_size
physical_id   = request.blocks[logical_block].block_id
```

The final storage location is:

```text
[layer_idx, physical_id, :, block_offset, :]
```

For example, with `block_size=2`:

```text
position 0 -> request block 0, offset 0
position 1 -> request block 0, offset 1
position 2 -> request block 1, offset 0
position 3 -> request block 1, offset 1
```

The physical IDs need not be adjacent. The request block list restores logical
order.

### Append and read

`PagedLayerKV.append`:

1. validates shape, batch size, KV-head count, dtype, device, and head size;
2. computes the logical start and end positions;
3. checks that the request already owns enough blocks;
4. writes each new token to its translated physical slot;
5. advances that layer's logical length;
6. calls `read` and returns the gathered dense history.

`PagedLayerKV.read` walks logical positions in order, translates each position,
loads K/V from physical storage, and stacks the values into:

```text
[1, num_kv_heads, logical_sequence_length, head_dim]
```

The current SDPA backend therefore receives dense K/V even when persistent
storage is paged. This gather is a correctness-first bridge, not direct paged
attention.

## Blocks and lifecycle

Two classes cooperate without storing the same kind of state.

### `BlockPool`

[`engine/block_pool.py`](../src/barellm/engine/block_pool.py) owns the finite set
of physical block IDs and tracks which IDs are free. It does not own K/V tensors,
request lengths, or model state.

### `KVCacheManager`

[`engine/kv_cache_manager.py`](../src/barellm/engine/kv_cache_manager.py) owns the
request lifecycle and the request-to-block assignment used for allocation. For
a request of length `T`, it requires:

```text
ceil(T / block_size)
```

blocks. `allocate_request` allocates only the difference between the required
count and the blocks already assigned. It then registers the complete block
list with `PagedKVCache`.

If the pool lacks capacity, allocation returns `False`. During admission the
engine leaves later requests waiting; during decode it skips requests that
cannot obtain their next block. Storage also checks capacity defensively and
raises if an append would exceed the registered block list.

`free_request`:

```text
removes manager ownership
  -> returns physical IDs to BlockPool
  -> unregisters the request from PagedKVCache
```

Physical tensor memory is not zeroed on free. A reused request starts with fresh
zero logical lengths and can only read positions it has written, so stale block
contents are outside its logical cache.

## Prefill and decode timeline

### Prefill

For a new request:

```text
Scheduler selects request
  -> KVCacheManager allocates blocks for prompt length
  -> PagedKVCache registers request and zero layer lengths
  -> Engine calls model with all prompt tokens and the request cache view
  -> every layer appends prompt K/V into its physical blocks
  -> attention gathers dense history and runs causal SDPA
  -> engine samples one token and appends its ID to Request.token_ids
```

The sampled token ID is now part of the request sequence, but its K/V is not in
the cache yet. It becomes the input to the first decode call.

### Decode

Before decoding, the manager ensures the request owns enough blocks for its
current `seq_len`. The engine then sends only the latest token:

```text
input_ids:    [batch, 1]
position_ids: [batch, 1]
```

Each layer appends that token's K/V. The model returns logits used to sample the
next token ID, which is appended to the request and becomes the following decode
input.

This creates the useful invariant:

```text
before a decode forward:
request token count = cached token count + 1

after that forward:
request token count = cached token count

after sampling:
request token count = cached token count + 1
```

## Batched decode adapter

[`engine/batched_kv_cache.py`](../src/barellm/engine/batched_kv_cache.py) does not
own persistent K/V storage. It is a temporary model-facing adapter over a list
of per-request caches.

For each layer it:

1. splits batched new K/V into batch-size-one rows;
2. appends each row to its request's underlying layer cache;
3. gathers the complete K/V history returned by each request;
4. pads histories to the longest sequence;
5. stacks them into a dense batch;
6. creates a boolean valid-key mask.

For final lengths `[3, 5]`, the mask is:

```text
request A: [True, True, True, False, False]
request B: [True, True, True, True,  True]
```

Its SDPA shape is `[batch, 1, query_length, max_key_length]`. The current engine
uses this adapter only for one-token decode, where `query_length=1`. The mask is
a padding mask for that supported path; it is not a causal mask for multi-token
cached batching.

## Ownership summary

| Component | Owns | Does not own |
| --- | --- | --- |
| `KVCache` / `LayerKV` | Interface contract | Storage or allocation |
| Attention | Q/K/V math and layer selection | Blocks or request lifetime |
| `ContiguousKVCache` | Dense per-layer K/V | Shared capacity management |
| `BlockPool` | Free and allocated physical IDs | K/V tensor contents |
| `KVCacheManager` | Request allocation lifecycle | Attention math |
| `PagedKVCache` | Physical K/V tensors, request page views, layer lengths | Scheduling decisions |
| `BatchKVCache` | Temporary routing, padding, and mask construction | Persistent K/V storage |
| `Engine` | Prefill/decode orchestration | K/V layout details |

## Current constraints

- `PagedLayerKV` accepts batch size one; `BatchKVCache` composes request views.
- Engine batching is one-token decode only.
- Paged reads gather dense tensors before SDPA.
- Blocks are allocated on demand but requests are not preempted or evicted.
- Blocks are not shared between requests and have no reference counts.
- There is no prefix cache or sliding-window truncation.
- K/V storage is not quantized.
- `KVCacheManager` currently manages `PagedKVCache` specifically rather than an
  abstract storage allocator.

These constraints are explicit boundaries of the current implementation, not
guarantees of the protocol for future backends.

## Verification

Cache behavior is covered at several levels:

- `tests/test_contiguous_kv_cache.py`: append/read contract and validation;
- `tests/test_paged_kv_cache.py`: block boundaries, request isolation, and
  insufficient allocation;
- `tests/test_kv_cache_manager.py`: growth, free, and block reuse;
- `tests/test_cache_equivalence.py`: contiguous cached MHA/GQA/MQA equivalence;
- `tests/test_qwen3.py`: multi-step paged MHA/GQA/MQA equivalence;
- `tests/test_batched_kv_cache.py`: padding masks and real batched paged decode;
- `tests/test_engine.py`: request admission, mixed lengths, cleanup, and decode
  behavior under exhausted capacity.

The main semantic checks are:

```text
cached logits ~= full recomputation logits
batched logits ~= independent request logits
read(write(logical K/V)) == logical K/V
```

See [SDPA.md](SDPA.md) for the attention and mask behavior consuming these
caches.
