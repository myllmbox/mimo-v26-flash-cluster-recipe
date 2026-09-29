# MiMo-V2.6-Flash on two DGX Sparks

Xiaomi's [MiMo-V2.6-Flash-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD) (309B MoE, 15B active) on two DGX
Sparks, TP=2 over RDMA, from the checkpoint as published. **49 tok/s single-stream, 202 tok/s at 16 streams (234 peak), 69 tok/s
peak with thinking on, a 1.22M-token KV pool, 300k context — and a warm boot serves in about 3 minutes.** Three commands.

## Measured performance (this exact stack, 2× DGX Spark, RDMA, MTP K=3, `vm.compaction_proactiveness=0`)

Thinking off, steady-state averages over 300-s windows. The prompt is a full ERP / supply-chain spec (SQL schemas, TypeScript,
an analysis table, load-test script and maths), a ~10k-token answer.

| concurrent requests | tok/s | peak | per-stream | acceptance |
|---|---|---|---|---|
| 1 | **49** | 62 | 49 | 2.83 |
| 2 | **69** | 86 | 35 | 2.94 |
| 4 | **90** | 112 | 22 | 2.85 |
| 8 | **134** | 163 | 17 | 3.09 |
| 16 | **202** | 234 | 13 | 3.25 |

Thinking on, one request (the pasture scene): **51 tok/s** average, 69 peak.

### Long context (one request)

Cold = a task after a fresh prompt of that size; hot = the next turn of the same conversation (prefix cache — an agent's next
step). Decode = 1,000 tokens after the prompt.

| prompt tokens | cold time to first token | hot time to first token | prefill tok/s | decode cold | decode hot |
|---|---|---|---|---|---|
| 1k | **0.69 s** | 0.28 s | 1,771 | 48.4 | 49.7 |
| 8k | 4.3 s | 0.29 s | 1,947 | 49.0 | 48.7 |
| 32k | 20 s | 0.35 s | 1,668 | 48.7 | 48.5 |
| 64k | 49 s | 0.41 s | 1,331 | 39.4 | 48.0 |
| 128k | 2 min 19 s | **0.57 s** | **947** | 44.7 | 38.8 |
| 192k | 4 min 27 s | 0.71 s | 737 | 42.0 | 36.4 |
| 256k | 7 min 15 s | 0.81 s | 602 | **40.0** | 34.8 |

## Quality (measured on this serve, thinking on)

| task | questions | score |
|---|---|---|
| HumanEval pass@1 | 164 | **95.7** |
| GSM8K exact match | 200 | **90.0** |
| IFEval prompt-level strict / instruction-level strict | 200 | **87.5** / 91.2 |
| MMLU-Pro | 200 | **80.4** |

## Quick start

```bash
git clone https://github.com/myllmbox/mimo-v26-flash-cluster-recipe.git
cd mimo-v26-flash-cluster-recipe
./run.sh        # first run: sets the cluster up (asks for the 2nd box), downloads ~178G, syncs it, serves on :8000
```

`./stop.sh` stops both boxes. `./view.sh` shows live stats plus the RDMA proof. Requirements: two DGX Sparks with docker +
the NVIDIA container runtime, connected by their ConnectX ports, ssh from the head to the worker (a password once —
`setup.sh` installs a key). With the weights on both boxes a boot serves in about 3 minutes.

**All model configuration lives in [`recipe.yaml`](recipe.yaml)** — image, weights, port, KV budget, every vLLM flag. The
cluster flags and the per-box NCCL/gloo pins come from `cluster.env` (written by `setup.sh`).

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

Thinking is on by default; send `"chat_template_kwargs": {"enable_thinking": false}` to turn it off per request.

## Tuning (recipe.yaml)

- **Drafter.** MTP K=3 on the checkpoint's three trained draft layers. K=2 is one line away: the same average, a lower peak,
  a little faster on long prose. The checkpoint's DFlash drafter is commented under it too.
- **KV pool.** `kv-cache-memory-bytes` is per box. 16 GB = 1.22M pooled tokens and leaves ~5 GB of host headroom on the head
  after graph capture — do not go higher.
- **Seats.** `max-num-seqs: 16`.
- **Loader.** `load-format: mbx` is ours (about a minute to load). `safetensors` is vLLM's stock loader: correct, but ~16 min
  and a slower serve. Never fastsafetensors or instanttensor on this checkpoint — the output is garbage from the first token.
- **Patches.** `patches` (server): optional vLLM patches from [`patches/`](patches/), off by default — e.g.
  `patches: hermes-chat` for the Hermes agent (by [@yume-arasaki](https://github.com/yume-arasaki)). Applied at launch
  over the image's files; the image itself is unchanged.

## Memory on a Spark

`run.sh` waits for both boxes to have their memory back and evicts its own checkpoint files from the page cache before launch
— no root needed. With root, `./tune-host.sh` sets `vm.compaction_proactiveness=0` on both boxes (worth ~10 % on a serve this
close to the memory edge). `./cache` is per box and deletable; the next boot rebuilds it.

## What's in the image

`myllmbox/mimo-v26-flash-cluster-vllm:v2` — `vllm/vllm-openai:nightly-ddd6fbca148a867aad1fcab7ec72f582b9977db4` (vLLM
0.30.1rc1, 2026-09-26; release 0.30.0 cannot serve this checkpoint) plus our prebuilt loader and kernel choices, split-KV
attention during speculative decoding, and all three trained MTP draft layers.

Digest: `sha256:5af49bd0c38923d1d2b3636aa7570b311ff55480f123207259b083cd9afde726`

v1 (the RL checkpoint, MTP K=1, image `:v1`) = `git checkout v1`.

## The full box

The same model also runs under [myllmbox](https://myllmbox.com) with a public HTTPS tunnel, dashboard, keepalive and
multi-model management — same image, same weights.

## License

Weights: MIT (XiaomiMiMo). Kit scripts: MIT.
