# Two-host gufo TP=2 server as systemd user units

Runs the gufo LLM server with tensor parallelism 2 across two Linux hosts,
entirely from systemd user units — no wrapper shell scripts:

- **rank 0 (head)**: serves the OpenAI-compatible HTTP API; unit
  `gufo@.service`, instance name = quantization (`q4`, `q8`, ...)
- **rank 1 (worker)**: GPU peer; unit `gufo-rank1@.service` on the second
  host. The head unit configures, starts and stops it over ssh, so a single
  `systemctl --user start gufo` brings the whole pair up.

The head host holds the whole configuration. Before it starts the worker, it
hands the worker's user manager the shared settings
(`systemctl --user set-environment`): its own RDMA-link address, image,
models, session counts and disk-cache budget. Both ranks therefore always
agree, and the worker host needs no environment files.

Each start also creates a new TP control token in the head's runtime
directory (`/run/user/<uid>/gufo.token`, a tmpfs, mode 0600) and writes it
through ssh's input into the same place on the worker host. The token is
never stored on disk and never appears in ssh's arguments; a worker that
restarts on its own reuses it, and the next head start replaces it.

Each rank runs `podman run --rm` (no `-d`) in the foreground; systemd owns
the process and the restart policy. The container output reaches the user
journal once, through the unit (`--log-driver none`), tagged `gufo-rank0` or
`gufo-rank1`.

## Units

| Unit                  | Host   | Purpose                                   |
|-----------------------|--------|-------------------------------------------|
| `gufo@.service`       | rank 0 | head server; instance = quantization      |
| `gufo.service`        | rank 0 | wrapper: `start gufo` → `gufo@q8`         |
| `gufo-rank1@.service` | rank 1 | worker; configured and started by the head |

None of the units has an `[Install]` section: they are started explicitly,
never at boot, so they do not claim the GPU before other local GPU services.
If another local user unit claims the GPU, add a drop-in
(`gufo@.service.d/local.conf` with `Conflicts=<unit>`).

## Prerequisites (both hosts)

- gufo checkout with a built `build/gpu-tp2/gufo` (the container mounts it
  read-only): `~/git/gufo`, or `GUFO_CHECKOUT` in `head.env`
- ROCm runtime image in podman (`GUFO_IMAGE`); [Containerfile](Containerfile)
  builds one, and the same image builds gufo:

  ```sh
  cd ~/git/gufo
  podman build -t localhost/gufo-tp2-dev:7.2.3 tools/systemd-user-service
  podman run --rm --security-opt label=disable --userns keep-id \
    -v "$PWD:/workspace/gufo" localhost/gufo-tp2-dev:7.2.3 \
    sh -c 'cmake --preset gpu-tp2 && cmake --build --preset gpu-tp2 --target gufo'
  ```
- the RDMA device nodes under `/dev/infiniband`; the units pass the whole
  directory, so gufo can pick any device with `--tp-rdma-device`
- the models in one directory at the same path on both hosts
  (`GUFO_MODEL_DIR`, default `/opt/models`; mounted read-only)
- on the head host: passwordless ssh to the worker host, and `rdma` and `ip`
  (iproute2) to find its RDMA-link address

## Install

```sh
# head host
cp 'gufo@.service' gufo.service ~/.config/systemd/user/
mkdir -p ~/.config/gufo
cp head.env.example ~/.config/gufo/head.env   # then adjust
systemctl --user daemon-reload

# worker host
cp gufo-rank1@.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

The worker connects to the head's address on the RDMA link. The head finds
it as the IPv4 address of its only RDMA netdev; with several, set
`GUFO_TP_NETDEV` or `GUFO_TP_ADDR` in `head.env`.

## Usage (all on the head host)

```sh
systemctl --user start gufo          # start pair (q8); returns in ~1 s
systemctl --user start gufo@q4       # or another quantization
systemctl --user stop gufo           # stop pair
systemctl --user status gufo@q8
journalctl --user -u gufo@q8 -f                          # rank 0 logs
ssh <worker-host> journalctl --user -u gufo-rank1@q8 -f  # rank 1 logs
```

`head.env` names one model per quantization (`GUFO_MODEL_Q8`,
`GUFO_MODEL_Q4`, ...) relative to `GUFO_MODEL_DIR`; the instance name selects
it, and the head hands it to the worker. Model load takes ~25 s per rank; the server answers
`http://<head>:8000/v1/models` ~1 minute after `start`.

## Disk cache

Both ranks keep their half of each continuation snapshot under
`<cache>/gufo-tp2/<quant>` on their own disk, so cached prompts survive a
restart. `GUFO_CACHE_DISK_BYTES` bounds each rank (default 64 GiB).
`<cache>/gufo` keeps the model files' identity digests, so a restart skips
hashing the full weights again (over a minute for Q8). `<cache>` is
`GUFO_CACHE_DIR` from `head.env`, or each host's `~/.cache`.

## Crash behavior

- A rank that loses its peer exits non-zero; `Restart=on-failure`
  (`RestartSec=5`) restarts it. A restart of the head also reconfigures and
  restarts the worker, so the pair converges.
- `podman run --rm` removes the container with the process, so no stale
  containers accumulate.
- Stopping `gufo` (or `gufo@<quant>`) also stops the worker via ssh
  (`ExecStopPost`, best effort); a stopped rank exits 0 and stays down.

## Notes

- The units' shell commands read their settings from the environment; the
  TP bootstrap (18515) and control (18516) ports must be reachable between
  the hosts on the RDMA link.
