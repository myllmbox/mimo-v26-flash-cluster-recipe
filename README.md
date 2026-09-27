# MiMo-V2.6-Flash-RL on two DGX Sparks

Xiaomi's [MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) (309B MoE, 15B active) on two DGX Sparks,
TP=2 over RDMA, the official checkpoint as published. **38 tok/s single-stream, 162 tok/s at 16 streams, a 1.45M-token KV pool,
300k context — and decode stays at 26 tok/s even with a 256k-token prompt.** Three commands.

## Measured performance (this exact stack, 2× DGX Spark, RDMA, MTP K=1, `vm.compaction_proactiveness=0`)

Thinking off, code prompt, steady-state averages (6 runs per rung up to 8 streams, 3 at 16).

| concurrent requests | tok/s | peak | per-stream | acceptance |
|---|---|---|---|---|
| 1 | **38** | 40 | 38 | 1.86 |
| 2 | **58** | 60 | 29 | 1.88 |
| 4 | **82** | 87 | 21 | 1.89 |
| 8 | **113** | 122 | 14 | 1.89 |
| 16 | **162** | 170 | 10 | 1.89 |

Thinking on, one request: **36 tok/s** average, 40 peak.

### Long context (one request)

Cold = fresh prompt; hot = the same prompt again (prefix cache — an agent's next turn). Decode = 200 tokens right after the
prompt: prose = a summary of it, code = a Python module.

| prompt tokens | cold time to first token | hot time to first token | prefill tok/s | decode prose | decode code |
|---|---|---|---|---|---|
| 1k | **0.63 s** | 0.29 s | 1,711 | 35.6 | 35.6 |
| 8k | 4.0 s | 0.32 s | 2,042 | 37.4 | 34.9 |
| 32k | 19 s | 0.39 s | 1,720 | 35.6 | 31.6 |
| 64k | 48 s | 0.49 s | 1,368 | 34.1 | 31.5 |
| 128k | 2 min 15 s | **0.66 s** | **970** | 30.3 | 31.0 |
| 192k | 4 min 22 s | 0.83 s | 751 | 27.3 | 27.0 |
| 256k | 7 min 9 s | 1.06 s | 611 | **26.4** | 22.8 |

## Quality (measured on this serve, thinking on)

| task | questions | score |
|---|---|---|
| HumanEval pass@1 | 164 | **94.5** |
| GSM8K exact match | 200 | **94.0** |
| IFEval prompt-level strict / instruction-level strict | 200 | **89.0** / 92.7 |
| MMLU-Pro | 200 | **79.9** |

## Quick start

```bash
git clone https://github.com/bilikaz/mimo-v26-flash-cluster-recipe.git
cd mimo-v26-flash-cluster-recipe
./run.sh        # first run: sets the cluster up (asks for the 2nd box), downloads ~178G, syncs it, serves on :8000
```

`./stop.sh` stops both boxes. `./view.sh` shows live stats plus the RDMA proof. Requirements: two DGX Sparks with docker +
the NVIDIA container runtime, connected by their ConnectX ports, ssh from the head to the worker (a password once —
`setup.sh` installs a key). With the weights on both boxes a boot reaches healthy in about 12 minutes.

**All model configuration lives in [`recipe.yaml`](recipe.yaml)** — image, weights, port, KV budget, every vLLM flag. The
cluster flags and the per-box NCCL/gloo pins come from `cluster.env` (written by `setup.sh`).

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "XiaomiMiMo/MiMo-V2.6-Flash-RL",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

Thinking is on by default; send `"chat_template_kwargs": {"enable_thinking": false}` to turn it off per request.

## Tuning (recipe.yaml)

- **Drafter.** MTP (`num_speculative_tokens: 1`) is the default. The checkpoint's DFlash drafter is one line away (commented
  under it): faster on code (63–68 tok/s single-stream), slower on prose and with thinking on.
- **KV pool.** `kv-cache-memory-bytes` is per box. 19 GB = 1.45M pooled tokens and leaves ~2 GB of host headroom on the head
  after graph capture — do not go higher.
- **Seats.** `max-num-seqs: 16`.
- **Loader.** Keep `load-format: safetensors` — the fast loaders (fastsafetensors, instanttensor) load this checkpoint
  incorrectly and the output is garbage from the first token.

## Memory on a Spark

`run.sh` waits for both boxes to have their memory back, evicts its own checkpoint files from the page cache before launch and
keeps evicting them during the ~10 min load (the plain loader would otherwise fill the cache) — no root needed. With root, `./tune-host.sh` sets
`vm.compaction_proactiveness=0` on both boxes (worth ~10 % on a serve this close to the memory edge).

## What's in the image

`myllmbox/mimo-v26-flash-cluster-vllm:v1` — `vllm/vllm-openai:nightly-ddd6fbca148a867aad1fcab7ec72f582b9977db4` (vLLM
0.30.1rc1, 2026-09-26; release 0.30.0 cannot serve this checkpoint) plus two readable patches: the loader page-cache drop, and split-KV attention during speculative decoding (stock vLLM
falls back to a kernel whose speed drops with context length; with the patch decode stays near its short-context speed).

Digest: `sha256:73708971f079a8b9d6a95c0b17a4eaf1ef795f8357c2f2d45110269a06297401`

## The full box

The same model also runs under [myllmbox](https://myllmbox.com) with a public HTTPS tunnel, dashboard, keepalive and
multi-model management — same image, same weights.

## License

Weights: MIT (XiaomiMiMo). Kit scripts and image patch: MIT.
