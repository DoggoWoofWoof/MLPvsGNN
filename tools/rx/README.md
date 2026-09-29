# rx — remote execution over ssh

`rx` runs a project's commands on another machine (here: a lab workstation with an
RTX 4500 Ada and 32 CPU threads), sends only the files that changed, keeps jobs
running through network outages, brings the outputs back, and shows every host and
job in one monitor.

It is two files and uses only the Python standard library:

| file | runs on | role |
|---|---|---|
| `rx.py` | your machine (Python 3.11+) | the CLI: setup, push, run, logs, wait, fetch, monitor, dashboard |
| `rx_agent.py` | the host (Python 3.8+) | answers one request per ssh session; supervises detached jobs |
| `test_rx.py` | your machine | `python -m pytest tools/rx/test_rx.py -q` (includes an end-to-end run on a local pseudo-host) |

Nothing is installed on the host beyond a standalone CPython and the agent file, both
under `~/rx`. No daemon, no open port: every interaction is one ssh session.

---

## Quick start

```sh
python tools/rx/rx.py setup gpu            # once per host: python, agent, capacity
python tools/rx/rx.py doctor --speed 32    # route, clock, GPU, disk, WSL, throughput
python tools/rx/rx.py env build mpr-cpu -f # once per rx.toml env change (detached job)
python tools/rx/rx.py run -- python scripts/foo.py --arg 1
python tools/rx/rx.py logs -f              # last job; survives disconnects
python tools/rx/rx.py wait last --fetch    # block, then bring outputs back
python tools/rx/rx.py monitor              # every host / project / job, live
```

On Windows `tools\rx\rx.cmd` is a shortcut for `python tools\rx\rx.py`.

`rx run -p gpu -- python train.py` uses the `[profiles.gpu]` table (CUDA env, 1 GPU).
`rx run --gpus 0.5 ...` shares one GPU between two jobs. `--cpus/--mem` reserve
resources; the host's scheduler queues jobs that do not fit yet.

---

## Configuration

### `rx.toml` at the project root

```toml
[project]
name = "mpr"          # namespace on the host: ~/rx/projects/mpr/
host = "gpu"          # default host (a name from `rx setup`)

[push]
include = ["git:tracked"]         # also: "git:untracked", or globs like "configs/**", "*.py"
exclude = ["legacy/**"]
max_mb = 512                       # refuse a push bigger than this without --force

[run]                              # defaults for every `rx run`
env = "mpr-cpu"                    # a [envs.*] table built with `rx env build`
cpus = 8                           # reserved AND exported as OMP/MKL/... thread counts
mem_gb = 16                        # reserved (scheduling); --mem-hard N sets a kill limit
outputs = ["outputs/**", "logs/**"]  # what counts as a job's outputs
env_vars = { PYTHONHASHSEED = "0" }

[profiles.gpu]                     # rx run -p gpu
env = "mpr-cu128"
gpus = 1

[envs.mpr-cpu]                     # a pinned environment, content-addressed by its spec
extra_index_urls = ["https://download.pytorch.org/whl/cpu"]
find_links = ["https://data.pyg.org/whl/torch-2.8.0+cpu.html"]
requirements = ["torch==2.8.0+cpu", "..."]
verify = ["import torch; print(torch.__version__)"]
```

`tools/rx/example.rx.toml` is a commented template for a new project.

### `~/.rx/` on your machine

* `hosts/<name>.json` — written by `rx setup`: ssh routes, host OS, interpreter and
  agent paths, stream count, compression level. Edit `routes` to add fallbacks.
* `projects.json` — projects seen, so `rx monitor` and `rx watch` find them.
* In each project, `.rx/` (git-ignored): `jobs.json` (jobs launched from here),
  `hashcache.json` (file hashes keyed by size+mtime), `receipts.json` (what `rx fetch` wrote).

---

## How it works

### Transport

Every request is **one ssh session** that runs `python rx_agent-<sha>.py rpc`: one JSON
line in, one JSON line out, optionally followed by a binary stream. The agent file name
carries its hash; when `rx.py` has a newer agent, it uploads it first (bootstrap through
`python -c` on stdin), so the two ends can never disagree about the protocol.

Transport failures (ssh exit 255, a cut stream) are retried with exponential backoff
(1 s → 30 s). A host can have several **routes** (`routes: [["gpu"], ["-o", "HostName=10.1.12.23", "gpu"]]`);
a failing route is skipped for two minutes.

Windows OpenSSH does not support connection multiplexing (`ControlMaster` sessions are
reset), so each request pays a full handshake, about 1 s over Tailscale. rx batches
work into few requests (one `plan`, a few parallel streams, one `launch`) to keep
this cheap.

### Transfers (push and fetch)

1. **Manifest.** The local side lists the declared files and hashes them (SHA-256,
   cached by size+mtime, so an unchanged tree costs a `stat` per file).
2. **Plan.** One request sends `[path, size, sha, mtime]` for every file. The host
   answers with what it already has (same sha), what it can **copy from another file
   with the same content** (renames, duplicates — no bytes on the wire), and what it
   needs, including a **resume offset** for any file an earlier transfer cut off.
3. **Streams.** Needed files are split into up to `streams` (default 4) groups of similar
   size, each sent over its own ssh session in parallel. Each 1 MiB chunk is
   zlib-compressed when that saves ≥10 %; after four chunks that do not compress,
   the file is sent raw (numpy, parquet and checkpoints are already dense).
4. **Landing.** A file is written to `name.rxpart` with a sidecar recording the promised
   sha, fsynced every 64 MiB, hashed while it arrives, and **renamed into place only if
   the size and sha match**. A corrupted or changed-during-read file is discarded, never
   installed. An interrupted file keeps its `.rxpart`; the next attempt re-hashes that
   prefix and continues from its end.
5. **Re-plan.** The loop repeats until nothing is needed, riding out disconnects for up
   to 30 minutes, so `rx push` after a drop simply resumes.

Fetch is the same protocol in reverse, with one extra rule: **rx never overwrites a
local file it did not write.** `.rx/receipts.json` records the size, mtime and sha of
everything `rx fetch` wrote; a local file that still matches its receipt may be
replaced by a newer output, any other differing file is reported as a conflict (exit
code 2) and left alone. `--overwrite` or `--into DIR` resolve conflicts explicitly.

`rx push --prune` deletes host files that rx pushed earlier and that are no longer
declared — but never a file whose content changed on the host.

### Jobs

`rx run` pushes, then sends one `launch` request. The host:

1. creates `~/rx/projects/<p>/jobs/<id>/` atomically (a temp dir renamed into place, so a
   retried launch with the same id is a no-op, never a duplicate);
2. starts a **supervisor** process fully detached from the ssh session;
3. returns. The ssh session ends; the job is independent of it.

The supervisor:

* **queues** the job under a host-wide lock until its cpus/memory/GPUs fit
  (`~/rx/host.json` holds capacity: all CPUs and memory minus a reserve for the OS and
  other users; GPUs from `nvidia-smi`). FIFO with backfill: a small job may overtake a
  big one that cannot fit, until the big one has waited `hold_after_s` (30 min).
  **Project priority** (`host.json`'s `priority` map, set with
  `rx capacity --priority PROJECT=N`; a project not in the map is 0) comes before FIFO:
  a queued job of a higher-priority project goes first however young it is, and while it
  cannot fit, lower-priority jobs may start only in what is left after its request is set
  aside, so they cannot starve it. Equal priorities keep FIFO, backfill and the hold. A
  job of a project below the map's top priority runs at **below-normal** OS priority
  (the Job Object's priority class, which every child inherits), so under CPU contention
  it gives way and otherwise uses the idle machine in full. Nothing running is ever
  stopped or re-prioritised: the policy applies to jobs as they are admitted, and a
  supervisor started by an older agent keeps the old rule until its job ends. Payloads
  run inside WSL are outside the Job Object's priority class;
* starts the command **suspended**, places it in a Windows **Job Object** (so every
  child process is accounted, limited by `--mem-hard`, and killed on cancel), then
  resumes it. On Linux it uses a process group;
* exports `CUDA_VISIBLE_DEVICES` (the assigned GPUs only; `-1` for CPU jobs),
  `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS = cpus`, `RX_JOB_ID`, `RX_JOB_DIR`, `RX_CPUS`, `RX_GPUS`
  (the assigned GPUs, empty for CPU jobs). Not an empty `CUDA_VISIBLE_DEVICES`: the Windows
  CUDA runtime reads an empty value as unset and shows every GPU, while
  `torch.cuda.device_count()` reads it as none, so a CPU job could still allocate on the
  GPU (measured on the host 2026-09-29: `""` → runtime 1 device, torch 0, allocation
  succeeds; `-1` → 0, 0, refused);
* writes `output.log` (stdout+stderr), `status.json`, and a **heartbeat** every 10 s with
  live CPU seconds, process count and memory;
* on exit, lists the job's **outputs**: files under the `outputs` globs modified since
  the job started, minus pushed inputs that did not change, each with its sha
  (`outputs.json`) — this is what `rx fetch JOB` brings back;
* keeps the machine awake while jobs run (`SetThreadExecutionState`).

States: `launching → queued → running → finishing → done | failed | cancelled`, plus
`error` (rx itself failed) and `lost` (derived: the heartbeat is older than 120 s or
older than the host's boot time, i.e. the supervisor died or the host rebooted).

### Detaching on Windows: the finding that shaped the design

Windows OpenSSH places each session's processes in a Job Object that is **killed when the
session ends** — `start /b`, `DETACHED_PROCESS`, `CREATE_BREAKAWAY_FROM_JOB` and
`Start-Process` all die with the connection. A process created through **WMI**
(`Win32_Process.Create`) is parented to the WMI service, outside the session, and
survives. rx launches supervisors that way (`RX_LAUNCH=wmi`, the Windows default;
`systemd-run --user` or `setsid` on Linux). This was verified on the lab host with a
process that kept writing after its ssh session was closed.

### Environments

`rx env build NAME` runs as a detached job (it takes minutes): copies the standalone
CPython into `~/rx/envs/NAME@<spec-hash>/`, pip-installs the pinned requirements, runs
the `verify` snippets, and writes `rx-freeze.txt`. The environment is addressed by the
hash of its spec, so editing `[envs.NAME]` in rx.toml builds a new directory and
`rx env ls` reports the old one as `STALE`. Jobs name environments with `--env`.
A CUDA environment whose `verify` touches the GPU needs `gpus = 1` in its table: the
build job then reserves the GPU (a build without one is a CPU job and sees no GPU).
`gpus` is not part of the hash.

The host's base interpreter is the python.org **NuGet CPython** (signed by the PSF,
same build as the laptop), installed by `rx setup` under `~/rx/python/<ver>/`. The
Microsoft Store Python cannot be launched from an ssh session (app-execution aliases
need an interactive logon), so rx never uses it.

---

## Commands

| command | does |
|---|---|
| `rx setup HOST [--route 'SSH ARGS']...` | detect OS, install Python (Windows), upload the agent, report capacity |
| `rx doctor [--speed MB]` | every route, clock skew, agent, CPU/memory/disk, GPUs, envs, WSL, throughput |
| `rx push [--dry-run] [--prune] [--inputs GLOB]` | sync declared files |
| `rx run [opts] -- CMD...` | push + launch; prints the job id |
| `rx ls [--all]` / `rx status [JOB] [--json]` | jobs with state, runtime, resources, last output line |
| `rx logs [JOB] [-f] [--tail KB]` | output; `-f` reconnects by itself and resumes at the byte offset |
| `rx wait JOB... [--fetch] [--into DIR]` | block until done; outages are waited out |
| `rx fetch [JOB] [--include/--exclude GLOB] [--into DIR] [--overwrite]` | outputs back |
| `rx fetch --glob 'outputs/x/**'` | any workspace files by pattern (e.g. while a job still runs) |
| `rx cancel JOB...` / `rx rerun JOB` | stop (the whole process tree) / launch the same spec again |
| `rx env build NAME [-f]` / `rx env ls` | environments |
| `rx exec -- CMD` | a short command right now (dies with the connection: not for real work) |
| `rx monitor [--once]` | terminal view of all hosts: CPU, memory, GPU util/memory/temp/power, jobs |
| `rx dashboard [--open]` | the same in a browser at `http://127.0.0.1:8765` (local only; cancel needs a per-run token) |
| `rx watch` | fetch jobs launched with `--fetch` as soon as they end, whenever the host is reachable |
| `rx capacity [--cpus N --mem GB --reserve-cpus N ...]` | show or set what the scheduler shares out |
| `rx capacity --priority mpr=10` | set a project's queue priority (repeatable or comma-separated; `=0` removes it); `rx monitor` shows the map |
| `rx gc [--days 14 --keep 30] [--dry-run]` | remove old job dirs, stale partial files, superseded envs |

`JOB` accepts a full id, any unique substring, or `last` (the default).

Run options: `-n NAME`, `-p PROFILE`, `--env`, `--cpus`, `--mem`, `--mem-hard`,
`--gpus` / `--gpu`, `--shell {exec,cmd,powershell,bash,wsl}`, `--cwd`, `--inputs GLOB`,
`--outputs GLOB`, `--var K=V`, `--timeout S`, `--no-push`, `-f`, `-w`, `--fetch`.

---

## Failure modes and recovery

| what happens | effect on the job | what to do |
|---|---|---|
| Wi-Fi drops, laptop sleeps, VPN reconnects | none: the job runs on | `rx logs -f` / `rx wait` reconnect by themselves; otherwise rerun the command later |
| Push interrupted | partial files kept as `.rxpart` | run the same command; it resumes at the byte |
| Fetch interrupted | partial files kept locally as `.rxpart` | run the same command |
| Laptop closed for days | none | `rx ls`, `rx fetch JOB`, or leave `rx watch` running |
| Job exceeds `--mem-hard` | the Job Object stops it; state `failed` | raise the limit or the reservation |
| Host rebooted | running jobs are `lost` (heartbeat older than boot) | `rx rerun JOB`; checkpoint long jobs to the outputs dir so a rerun can resume |
| Supervisor killed | `lost` after 120 s | `rx cancel` clears it; `rx rerun` |
| Queue never admits | `rx status` shows `waiting for ...` / `impossible: ...` | lower the request, or `rx capacity` |
| Queued behind another project | `rx status` shows `waiting for higher-priority job ...` | expected while that project's job waits; `rx capacity` shows the map |
| Output file edited locally | `rx fetch` reports a conflict, exit 2 | `--into DIR` or `--overwrite` |
| Agent updated | uploaded automatically on the next call | nothing |

Reboots are the one outage rx cannot hide: a process cannot survive its machine.
Long jobs should write checkpoints under their outputs directory.

---

## The network (measured 2026-09-26/28)

The laptop reaches the host over **Tailscale** (`ssh gpu` → `100.118.95.8`).

* On the campus Wi-Fi both machines sit behind the same public NAT (223.31.218.223)
  with no hairpin, and inter-VLAN traffic from Wi-Fi to the wired lab subnet
  (10.1.12.0/24) is filtered (ports 22/445/3389 and ICMP all time out; the host's own
  firewall allows port 22 on every profile). Tailscale therefore relays through DERP
  (Bengaluru): **≈1.2–1.3 MB/s per stream, 1.7 MB/s on 4 streams**, ~56 ms.
* From a home connection with UDP and an endpoint-independent NAT, Tailscale can go
  **direct** — typically 10–100× the relay rate. Check with `tailscale ping 100.118.95.8`
  ("via DERP" vs "via <ip:port>") and measure with `rx doctor --speed 32`.
* 2026-09-29, laptop on a home network where IPv4 UDP did not reach Tailscale's STUN
  (`tailscale netcheck`: IPv4 none, IPv6 yes) and the host IPv4-only: DERP again.
  Down ≈0.75 MB/s whatever the stream count (1–8); up 0.73 MB/s on one stream, 1.2 MB/s
  on 4, 1.27 on 8. zlib level 6 stays the best trade on 1 MiB chunks: level 9 is 0.5 %
  smaller on code (7 % on text logs) but 2.6–3.8× slower, which caps a text fetch below
  level 6's rate; level 1 is 22–50 % larger. So `streams = 4`, `compress_level = 6`
  (the defaults) until a direct path exists.
* If the laptop is ever on the wired lab network, add a LAN route:
  `rx setup gpu --route gpu --route '-o HostName=10.1.12.23 gpu'`, preferring the LAN.

What this means for work placement: **move code, not data.** A push of the tracked tree
is ~11 MB; results (metrics JSON, logs) are small. Large inputs (datasets, embedding
caches) should be fetched or built on the host once, not pushed repeatedly; `max_mb`
stops accidental bulk pushes.

---

## Validated on the lab host (2026-09-29)

* `rx setup gpu`: Windows 11, NuGet CPython 3.13.5, 32 CPUs, 127.7 GB, RTX 4500 Ada 24 GB.
* Detach: a 90 s job kept running after its launch session closed and after a `logs -f`
  session was killed mid-stream; it ended rc=0 with all 90 ticks, and `wait --fetch`
  brought them back.
* Resume: a 12 MB push killed half-way re-sent only the missing 6.0 MB; the host's
  sha256 matched the laptop's.
* Environments: `mpr-cpu` and `mpr-cu128` built (2.8.0+cu128 on the RTX 4500 Ada). In a
  `-p gpu` job: fp32 matmul ≈21 TFLOP/s (no warm-up), `torch_scatter` and PyG `GATConv`
  on `cuda:0`. A CPU job in the same environment sees no GPU (`CUDA_VISIBLE_DEVICES=-1`,
  allocation refused).
* The repo's tests on the host (`mpr-cpu`, 8 CPUs, `--continue-on-collection-errors`):
  6,554 passed in 2 min (8,759 locally in 6 min). Every test that passes locally but not
  there needs something rx deliberately leaves behind: `outputs/` (never pushed), the git
  history (the workspace holds files only) or the Modal SDK (not installed on the host).
* WSL, after the account owner ran `wsl_setup.ps1` (every check ok: 94 GB visible under
  the 96 GB limit, systemd running, driver 596.71): `--shell wsl` jobs run as `student2`
  in `/mnt/c/Users/Student2/rx/projects/mpr/ws`, with the `RX_*` and thread variables
  forwarded; their outputs are written with Linux modes and fetched back. The WSL CUDA
  driver finds 1 device in a `--gpu` job and none in a CPU job (`cuInit` returns 100,
  no device). `nproc` inside a job reports the job's cpus: it honours `OMP_NUM_THREADS`.

---

## WSL on the host

Ubuntu 24.04 is installed under `~/rx/wsl/ubuntu-24.04` (WSL 2). rx jobs can run in it
with `--shell wsl` (`wsl.exe -d Ubuntu-24.04 --cd <ws> --exec bash -lc CMD`; the
workspace is the Windows directory seen through `/mnt/c`, and `WSLENV` forwards
`RX_*`, `CUDA_VISIBLE_DEVICES` and the thread variables). The configuration is in
`%USERPROFILE%\.wslconfig` (memory, processors, swap) and `/etc/wsl.conf` (systemd,
default user). CUDA inside WSL uses the Windows driver; no Linux driver is installed.

`tools/rx/wsl_setup.ps1` makes that configuration once: a Linux user named after the
Windows account with passwordless sudo (checked by `visudo` first), `/etc/wsl.conf`
(systemd, that default user, `appendWindowsPath=false`, `metadata` on `/mnt/c`), and
`.wslconfig` (96 GB, 32 processors, 32 GB swap, `autoMemoryReclaim=dropCache`,
`sparseVhd=true`). It then runs `wsl --shutdown` (this Windows account's distros
only) and checks every setting from inside the distro, ending with `WSL SETUP OK`.
Those are system and security settings, so **the account owner runs it**, not rx:

```sh
python tools/rx/rx.py push
ssh gpu "powershell -NoProfile -ExecutionPolicy Bypass -File rx\projects\mpr\ws\tools\rx\wsl_setup.ps1"
```

Native Windows is the default for rx jobs: the Job Object accounting, the GPU and the
file system are all first-class there, and `/mnt/c` I/O from WSL is slow for many small
files. Use WSL for tools that only exist on Linux.

---

## Security

* rx uses your existing ssh key and `~/.ssh/config`; it stores no credentials, and puts
  none on the host. **Do not copy API tokens (Modal, Hugging Face, cloud) to a shared
  lab machine.** `--var` values are written into the job's `spec.json` on the host, so
  they are not for secrets either: keep any step that needs a token on your machine.
* The agent executes only what the CLI sends through your ssh session. Paths from the
  wire are validated (no absolute paths, `..`, drive letters, reserved names).
* The dashboard binds `127.0.0.1`, checks the `Host` header, and requires a random
  per-run token for the only state-changing call (cancel).
* The host is shared: the default capacity leaves a reserve for other users
  (1/16 of the CPUs, 6 % of memory), and `rx capacity` can lower it further.

---

## Porting to another project or host

1. Copy `tools/rx/` (three files) into the project, or keep one copy and call it by path.
2. Write an `rx.toml` from `example.rx.toml`: name, host, what to push, outputs, envs.
3. `rx setup NAME` for each new host (any ssh destination; Linux hosts need `python3`).
4. `rx env build ENV`, then `rx run`.

Hosts are shared across projects: one `~/rx` per host, one namespace per project.

---

## This repository's rule for scientific work

rx is **systems infrastructure**. In this repository, running a scientific stage on the
remote host — on its CPU or its GPU — is a *placement* decision that must be filed
before any number is read from it, as `configs/universal_v2.yaml#later_stages.compute_placement`
requires. The **CPU-GPU equivalence test** it asks for is
`configs/cpu_gpu_equivalence.yaml`. It ran on 2026-09-29 (`docs/CPU_GPU_EQUIVALENCE.md`):

| placement | verdict | what it means |
|---|---|---|
| `host_cpu_t8` (`mpr-cpu@31803e6457ab`, 8 threads) | `EQUIVALENT_WITHIN_TOLERANCE` | nameable |
| `host_gpu_det` (`mpr-cu128@62fc45e9e1ba`, float32, deterministic) | `NOT_EQUIVALENT` (2 cells) | score level: barred |
| `host_gpu_default` (same, determinism off) | `NOT_EQUIVALENT` (245 cells) | barred |

The equivalence file's fail line names a new declaration as the way back. That
declaration is `configs/gpu_task_qualification.yaml`, which ran on 2026-09-29
(`docs/GPU_TASK_QUALIFICATION.md`). It used a fresh probe and read at the task level:
top-k sets, R@5, FullCov@5 and Hit@1.

| placement | verdict | what it means |
|---|---|---|
| `host_gpu_det` (the settings above) | `TASK_EQUIVALENT` and `TRAINING_REPRODUCIBLE` | nameable as a **host-native** placement |

A host-native stage runs every arm it compares on the host GPU: the new arm and every
baseline, control and comparator, from one commit and on byte-verified inputs. A
number from the host GPU is never compared with a laptop number. The one exception is
a task-level evaluation of the same frozen checkpoint. The rules are in the file's
`host_native_protocol`.

A placement that passed is usable only through a dated authorization block of a
declared stage that quotes the tested settings and runs the file's mirror
verification first. A barred placement reopens only through a new arm of the same
test or a new declaration. Everything else on the host is tests, environment builds,
and systems validation. `docs/GPU_HOST_HANDOVER.md` is the operator's guide: rules,
layout, recipes and quirks for anyone picking the host up.
