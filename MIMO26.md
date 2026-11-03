# MiMo-V2.6-Flash-RL on gufo

This tree is [gufo](https://github.com/gufo-org/gufo) 0.15.0 plus the **mimo2**
port: Xiaomi MiMo-V2.6-Flash-RL — a 309B-parameter MoE with 15B active per token
and MXFP4 experts — served across two 128 GB Strix Halo boards (gfx1151) with
tensor parallelism over USB4/TB4 RDMA.

The operational layer (launcher, systemd supervisor, benchmark suite, runbook)
lives in [gufo-tp2-mimo26](https://github.com/kaka86mm/gufo-tp2-mimo26).
The USB4 write-striping transport is
[gufo-tp2-tb4-rdma](https://github.com/kaka86mm/gufo-tp2-tb4-rdma) plus
[neuhaus/donnerkeule](https://github.com/neuhaus/donnerkeule).

## Pinned build

- Upstream base: gufo `v0.15.0` (`89d6185`, 2026-10-13).
- 17 port commits plus this release commit (`git log --oneline 89d6185..HEAD`).
- Measured production binary:
  `96e5afcc8d56953f9984e6a4f4e6f79e` (303,330,824 bytes; also archived as
  `ac82b1ae`), built as `RelWithDebInfo` with ROCm 7.2.4 on gfx1151.
- A rebuild in a different directory is not byte-identical (debug paths are
  embedded). The source tree is the tree the measured binary was built from.
- Verified from a clean clone of this tag: the two flags above build a
  303,327,840-byte `gufo` that runs (`gufo --version`); its md5 differs only
  because the build directory is embedded in `RelWithDebInfo`.

Build:

```sh
sudo apt install build-essential cmake ninja-build pkg-config \
  libicu-dev libcurl4-openssl-dev libssl-dev libpng-dev libjpeg-dev \
  libwebp-dev libsndfile1-dev libsoxr-dev libavformat-dev libavcodec-dev \
  libavutil-dev libswscale-dev ffmpeg libibverbs-dev \
  hipblas-dev hipblaslt-dev rocblas-dev hipcub-dev rocprim-dev rocwmma-dev

cmake --preset release -DGUFO_ENABLE_TP2_RDMA=ON -DGUFO_MIMO2_HIP=ON
cmake --build --preset release --parallel 8
./build/release/gufo diagnose
```

Both flags are required: `GUFO_MIMO2_HIP` defaults to OFF, and without it the link fails on every `mimo2` symbol. The measured build used ROCm 7.2.4 at `/opt/rocm-7.2.4`.

## Model files

From [`ggml-org/MiMo-V2.6-Flash-RL-GGUF`](https://huggingface.co/ggml-org/MiMo-V2.6-Flash-RL-GGUF)
(MIT; the weights are not included here):

```sh
hf download ggml-org/MiMo-V2.6-Flash-RL-GGUF \
  --include "MiMo-V2.6-Flash-RL-MXFP4-0000*-of-00002.gguf" \
            "mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf" \
            "mmproj-MiMo-V2.6-Flash-RL-BF16.gguf" \
  --local-dir ~/models/mimo26-flash-rl
```

Expected sizes: 5,948,640 B (shard 1, metadata only) +
167,363,516,512 B (shard 2, the trunk), 2,383,728,864 B (MTP Q8_0), and
2,748,509,792 B (mmproj BF16; only needed for image input).

## Dense sidecar

The Q8_0 dense attention projections are requantized to Q4_0 in a sidecar GGUF
that the loader grafts over the originals (`GUFO_MIMO_DENSE_SC`). Regenerate it
with the included tool:

```sh
pip install gguf numpy
python3 tools/mimo2_requant_dense.py \
  --src ~/models/mimo26-flash-rl/MiMo-V2.6-Flash-RL-MXFP4-00002-of-00002.gguf \
  --out ~/models/mimo26-flash-rl/dense-full-q4_0.gguf \
  --type q4_0 --head q4_0 \
  --mtp ~/models/mimo26-flash-rl/mtp-MiMo-V2.6-Flash-RL-Q8_0.gguf
```

The production file is `87060794b099d820a08eb3b6e049a307`
(3,541,900,640 bytes). Without `GUFO_MIMO_DENSE_SC` the engine runs, but reads
the Q8_0 originals and is slower.

## Serving

Use the launcher in the ops repo. The production environment is:

```sh
GUFO_NO_GRAPH=1
GUFO_MIMO_DENSE_SC=$HOME/models/mimo26-flash-rl/dense-full-q4_0.gguf
GPU_MAX_HW_QUEUES=1
GUFO_AR_F16=1
GUFO_TOOLS_FREEFORM=1
GUFO_MTP_COST_SCALE=0.80
GUFO_MTP_ACCEPT_DECAY=0.9
GUFO_MTP_RETRY_TOKENS=4
HIP_VISIBLE_DEVICES=0
LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib
```

Each rank runs:

```sh
gufo serve llm --model <shard1> --speculative mtp --mtp-model <mtp> \
  --tp-world-size 2 --tp-rank {0,1} \
  --tp-bootstrap-port 18525 --tp-control-port 18526 \
  --tp-control-token <secret> --tp-rdma-device usb4_rdma0 \
  --sessions 6 --context 204800 --prefill-chunk 2048 \
  --max-pending-per-client 16 \
  --cache-disk <dir> --cache-disk-bytes 30064771072 \
  --cache-ram-bytes 1073741824 --cache-disk-staging-bytes 3221225472
```

Rank 0 listens for the peer; rank 1 passes `--tp-bootstrap-host <rank0-ip>`.
`--cache-disk` must be a real directory owned by the engine user; a symlink is
rejected with `continuation disk directory is not a private owned directory`.

Graph capture is deliberately off: measured −40%, because replay kernels are
slower than eager on this workload. The EXL3 sources under
`src/models/mimo2/kernels/rocm/exl3_moe.*` are carried for reference; the
serving path in this tree does not dispatch them.

## What was measured

Two 128 GB boards, six 200K sessions: single stream 29.2–30.1 t/s at 1K,
six-way 85.5–86 t/s at 1K, cold prefill 682 t/s at 8K down to 238 t/s at 150K,
and a 193K prefix restored from disk in 7.8 s after a restart. Full tables,
dead ends, and methodology:
[gufo-tp2-mimo26](https://github.com/kaka86mm/gufo-tp2-mimo26).

## License

gufo is MIT (see `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`). The MiMo port
and `tools/mimo2_requant_dense.py` are released under the same MIT terms.
