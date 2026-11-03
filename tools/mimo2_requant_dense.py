#!/usr/bin/env python3
"""Requantize the dense attention projections of the MiMo MXFP4 artifact into
a sidecar GGUF that the engine grafts over the originals.

`blk.*.attn_qkv.weight` + `blk.*.attn_output.weight` ship as Q8_0 (1.0625
B/w) and are ~2.4 GB/rank/token of the decode byte budget, read cold from
DRAM every token. The vendored dense_vec / mmq entries already accept Q4_0
and Q4_K (0.5625 B/w, standard ggml block layouts -- no repack), and
ops/dense_q_probe.hip.cpp measured 1.6-1.8x per call at decode widths.

  usage: mimo2_requant_dense.py --src SRC.gguf --out SC.gguf --type q4_0|q4_K
                                 [--head TYPE] [--mtp MTP.gguf]
  the sidecar carries the same tensor names and logical shapes (dims
  [cols, rows]); the loader keeps its TP row-splitting, which is byte-generic.

Every tensor's requantization error is printed (relative RMS vs the source
Q8_0 dequant). The block layouts follow ggml exactly: Q4_0 packs element j in
the low nibble and j+16 in the high one; Q4_K packs 6-bit scales/mins in the
12-byte scales field (top 2 bits of the upper four live in the lower bytes)
and low/high nibbles per 64-value group.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import gguf

QK8_0 = 32
QK_K = 256


def dequant_q8_0(raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    nb = cols // QK8_0
    blk = raw.reshape(rows, nb, 34)
    d = blk[:, :, 0:2].copy().view(np.float16).astype(np.float32)[..., 0]
    qs = blk[:, :, 2:34].copy().view(np.int8).astype(np.float32)
    return (d[:, :, None] * qs).reshape(rows, cols)


def quant_q4_0(x: np.ndarray) -> np.ndarray:
    rows, cols = x.shape
    nb = cols // QK8_0
    xb = x.reshape(rows, nb, QK8_0)
    # ggml: d = max_signed / -8, then q = (int8_t)(x*id + 8.5) truncated.
    idx = np.abs(xb).argmax(-1)
    vmax = np.take_along_axis(xb, idx[:, :, None], -1)[:, :, 0]
    d = vmax / -8.0
    # A whole-zero block has d = 0; ggml stores it as-is and dequantizes to 0.
    idv = np.where(d == 0.0, 0.0, 1.0 / np.where(d == 0.0, 1.0, d))
    q = np.minimum(15, np.floor(xb * idv[:, :, None] + 8.5)).astype(np.uint8)
    out = np.zeros((rows, nb, 18), dtype=np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(rows, nb, 2)
    out[:, :, 2:18] = q[:, :, 0:16] | (q[:, :, 16:32] << 4)
    return out.reshape(-1)


def _k4_pack_scales(sc: np.ndarray, mn: np.ndarray) -> np.ndarray:
    """Pack 8 six-bit scales + 8 six-bit mins into ggml's 12 scales bytes."""
    s = sc.astype(np.uint16)
    m = mn.astype(np.uint16)
    b = np.zeros((sc.shape[0], sc.shape[1], 12), dtype=np.uint8)
    b[:, :, 0:4] = (s[:, :, 0:4] & 63) | (((s[:, :, 4:8] >> 4) & 3) << 6)
    b[:, :, 4:8] = (m[:, :, 0:4] & 63) | (((m[:, :, 4:8] >> 4) & 3) << 6)
    b[:, :, 8:12] = (s[:, :, 4:8] & 15) | ((m[:, :, 4:8] & 15) << 4)
    return b


def quant_q4_K(x: np.ndarray) -> np.ndarray:
    rows, cols = x.shape
    nb = cols // QK_K
    xb = x.reshape(rows, nb, 8, 32)  # 8 sub-blocks of 32 per super-block
    lo = xb.min(-1)
    hi = xb.max(-1)
    s = (hi - lo) / 15.0
    dmax = np.maximum(s.max(-1, keepdims=True), 1e-8)
    d = dmax / 63.0
    sc = np.clip(np.round(s / d), 0, 63)
    off = np.maximum(-lo, 0.0)
    mmax = np.maximum(off.max(-1, keepdims=True), 1e-8)
    dmin = mmax / 63.0
    mn = np.clip(np.round(off / dmin), 0, 63)
    sd = d * sc
    od = dmin * mn
    # Constant sub-blocks (hi == lo) carry scale 0; they decode to -dmin*mn,
    # same convention as ggml's quantizer (which also zeroes them).
    q = np.where((sd > 0)[..., None],
                 np.clip(np.round((xb + od[:, :, :, None]) /
                                  np.maximum(sd, 1e-12)[..., None]),
                         0, 15),
                 0.0).astype(np.uint8)
    out = np.zeros((rows, nb, 144), dtype=np.uint8)
    out[:, :, 0:2] = d.astype(np.float16).view(np.uint8).reshape(rows, nb, 2)
    out[:, :, 2:4] = dmin.astype(np.float16).view(np.uint8).reshape(rows, nb, 2)
    out[:, :, 4:16] = _k4_pack_scales(sc, mn)
    qs = out[:, :, 16:144].reshape(rows, nb, 4, 32)
    lo_nib = q.reshape(rows, nb, 4, 2, 32)[:, :, :, 0, :]
    hi_nib = q.reshape(rows, nb, 4, 2, 32)[:, :, :, 1, :]
    qs[:, :, :, :] = lo_nib | (hi_nib << 4)
    return out.reshape(-1)


def refdequant_q4_0(raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    nb = cols // QK8_0
    blk = raw.reshape(rows, nb, 18)
    d = blk[:, :, 0:2].copy().view(np.float16).astype(np.float32)[..., 0]
    qs = blk[:, :, 2:18]
    lo = (qs & 0xF).astype(np.float32) - 8.0
    hi = (qs >> 4).astype(np.float32) - 8.0
    return (d[:, :, None] * np.concatenate([lo, hi], axis=-1)).reshape(rows, cols)


def refdequant_q4_K(raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    nb = cols // QK_K
    blk = raw.reshape(rows, nb, 144)
    d = blk[:, :, 0:2].copy().view(np.float16).astype(np.float32)[..., 0]
    dmin = blk[:, :, 2:4].copy().view(np.float16).astype(np.float32)[..., 0]
    sb = blk[:, :, 4:16].astype(np.uint16)
    s = np.zeros((rows, nb, 8), dtype=np.uint16)
    m = np.zeros((rows, nb, 8), dtype=np.uint16)
    s[:, :, 0:4] = sb[:, :, 0:4] & 63
    m[:, :, 0:4] = sb[:, :, 4:8] & 63
    s[:, :, 4:8] = (sb[:, :, 8:12] & 15) | (((sb[:, :, 0:4] >> 6) & 3) << 4)
    m[:, :, 4:8] = ((sb[:, :, 8:12] >> 4) & 15) | (((sb[:, :, 4:8] >> 6) & 3) << 4)
    qs = blk[:, :, 16:144].reshape(rows, nb, 4, 32)
    lo = (qs & 0xF).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    q = np.concatenate([lo, hi], axis=-1)  # (rows, nb, 4, 64) -> sub-blocks 2g,2g+1
    q = q.reshape(rows, nb, 8, 32)
    y = d[:, :, None, None] * s[:, :, :, None] * q - dmin[:, :, None, None] * m[:, :, :, None]
    return y.reshape(rows, cols)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", required=True, choices=["q4_0", "q4_K"])
    ap.add_argument("--no-ffn0", action="store_true",
                    help="skip blk.0.ffn_{gate,up,down} (default: include)")
    ap.add_argument("--head", default="", choices=["", "q4_0", "q4_K"],
                    help="also requantize output.weight (the lm head, read "
                         "once per token for the trunk and again per draft "
                         "pass) at this type")
    ap.add_argument("--mtp", default="",
                    help="MTP sidecar GGUF: also requantize its blk.* dense "
                         "tensors (the draft network's own weights)")
    args = ap.parse_args()

    QUANT = {"q4_0": (quant_q4_0, refdequant_q4_0, gguf.GGMLQuantizationType.Q4_0),
             "q4_K": (quant_q4_K, refdequant_q4_K, gguf.GGMLQuantizationType.Q4_K)}
    targets = []  # (tensor, (qfn, ref_fn, qtype))
    reader = gguf.GGUFReader(args.src)
    for t in reader.tensors:
        n = t.name
        if ".attn_qkv.weight" in n or ".attn_output.weight" in n:
            targets.append((t, QUANT[args.type]))
        elif not args.no_ffn0 and n in ("blk.0.ffn_gate.weight",
                                         "blk.0.ffn_up.weight",
                                         "blk.0.ffn_down.weight"):
            targets.append((t, QUANT[args.type]))
    if args.head:
        for t in reader.tensors:
            if t.name == "output.weight":
                targets.append((t, QUANT[args.head]))
    if args.mtp:
        # The draft network's own blocks (blk.48.. in the sidecar's own
        # numbering). Its top-level output.weight/token_embd are unused by the
        # loader (the drafter falls back to the trunk head), so they are left
        # alone -- and the name would collide with the trunk's entry.
        mtp_reader = gguf.GGUFReader(args.mtp)
        for t in mtp_reader.tensors:
            if ".attn_qkv.weight" in t.name or ".attn_output.weight" in t.name \
                    or ".ffn_gate.weight" in t.name or ".ffn_up.weight" in t.name \
                    or ".ffn_down.weight" in t.name \
                    or ".nextn.eh_proj.weight" in t.name:
                targets.append((t, QUANT[args.type]))
    if not targets:
        print("no target tensors found", file=sys.stderr)
        return 1

    writer = gguf.GGUFWriter(args.out, arch="mimo2")
    total_in = total_out = 0
    stats = []
    for t, (qfn, ref_fn, qtype) in targets:
        if t.tensor_type != gguf.GGMLQuantizationType.Q8_0:
            print(f"!! {t.name}: source type {t.tensor_type} is not Q8_0; skipped")
            continue
        rows, cols = int(t.shape[1]), int(t.shape[0])
        raw = np.asarray(t.data).reshape(-1)
        x = dequant_q8_0(raw, rows, cols)
        blob = qfn(x)
        back = ref_fn(blob, rows, cols)
        rel = float(np.sqrt(np.mean((back - x) ** 2)) / np.sqrt(np.mean(x ** 2)))
        mx = float(np.max(np.abs(back - x)))
        total_in += raw.nbytes
        total_out += blob.nbytes
        stats.append((t.name, rel, mx, raw.nbytes, blob.nbytes))
        # gguf-py takes the *byte* shape (rows, bytes_per_row) and derives the
        # logical dims itself; passing raw_shape makes it double-convert.
        bs, ts = gguf.GGML_QUANT_SIZES[qtype]
        writer.add_tensor(t.name, blob.reshape(rows, cols // bs * ts),
                          raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"{'tensor':34s} {'relRMS':>8s} {'maxAbs':>9s} {'in MB':>8s} {'out MB':>8s}")
    for name, rel, mx, nb_in, nb_out in stats:
        print(f"{name:34s} {rel:8.4f} {mx:9.4f} {nb_in/1e6:8.1f} {nb_out/1e6:8.1f}")
    print(f"total: {total_in/1e6:.1f} MB -> {total_out/1e6:.1f} MB "
          f"({total_in/max(total_out,1):.2f}x smaller), {len(stats)} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())