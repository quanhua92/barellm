import torch
from safetensors import safe_open

from barellm.config import DTYPE, MODEL_ID
from barellm.hub import download_model


def load_weights(
    model_id: str = MODEL_ID, device="cpu", dtype=DTYPE
) -> dict[str, torch.Tensor]:
    snapshot_dir = download_model(model_id)
    safetensors_files = sorted(snapshot_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files in {snapshot_dir}")
    weights: dict[str, torch.Tensor] = {}

    for path in safetensors_files:
        with safe_open(str(path), framework="pt", device=device) as f:
            for key in f.keys():
                if key in weights:
                    raise ValueError(f"Duplicate tensor key {key} across shards")
                weights[key] = f.get_tensor(key).to(dtype)
    return weights


if __name__ == "__main__":
    w = load_weights()

    print(f"\n=== Total tensors: {len(w)} ===")

    root_keys = sorted(k for k in w if ".layers." not in k)
    print(f"\n=== Root Keys ({len(root_keys)}) ===")
    for k in root_keys:
        t = w[k]
        print(f"    {k:<55} {str(list(t.shape)):>20} {t.dtype}")

    layer0_keys = sorted(k for k in w if ".layers.0." in k)
    print(f"\n=== Layer 0 Keys ({len(layer0_keys)}) ===")
    for k in layer0_keys:
        t = w[k]
        print(f"    {k:<55} {str(list(t.shape)):>20} {t.dtype}")

    from collections import Counter

    counter = Counter()
    for k in w:
        if ".layers." in k:
            layer_id = k.split(".layers.")[1].split(".")[0]
            counter[layer_id] += 1
    print("\n=== Per-layer tensor counts ===")
    print(counter)

    count = sum(counter.values()) + len(root_keys)
    print(
        f"   layers({sum(counter.values())}) + root({len(root_keys)}) "
        f"= {count} | total: {len(w)} | match: {count == len(w)}"
    )

    counts = set(counter.values())
    assert len(counts) == 1, f"Inconsistent layer counts: {counts}"
    print(f"   All layers have {counts.pop()} tensors")

    print("\n=== QK-norm presence ===")
    print(f"    q_norm: {any('.q_norm.' in k for k in w)}")
    print(f"    k_norm: {any('.k_norm.' in k for k in w)}")

    if "lm_head.weight" in w and "model.embed_tokens.weight" in w:
        tied = torch.equal(w["lm_head.weight"], w["model.embed_tokens.weight"])
        print(f"\n=== Tied Embeddings: {tied}")
