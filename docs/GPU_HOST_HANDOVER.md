# GPU host handover

This guide is for any Claude session, or any person, that wants to use the lab GPU machine from this repository. It covers what exists, the rules, how to drive the host, and what goes wrong. It was written on 2026-09-29, after the host was set up and validated. The tool's full manual is [`tools/rx/README.md`](../tools/rx/README.md); this file doesn't repeat it and points into it.

## Read this first

1. **Science runs on the host only where a placement test passed and a stage names it.** That covers fits, evaluations, selections and any number that enters a result. A stage can run there only after two things:
   - [`configs/cpu_gpu_equivalence.yaml`](../configs/cpu_gpu_equivalence.yaml) (host CPU) or [`configs/gpu_task_qualification.yaml`](../configs/gpu_task_qualification.yaml) (host GPU, host-native stages only) has run and passed for that placement;
   - a dated authorization block of the stage names that placement.

   The test ran on 2026-09-29 ([`docs/CPU_GPU_EQUIVALENCE.md`](CPU_GPU_EQUIVALENCE.md)). **Host CPU** (`mpr-cpu@31803e6457ab`, 8 threads): `EQUIVALENT_WITHIN_TOLERANCE`, so it is nameable. **Host GPU**: `NOT_EQUIVALENT` at score level in both modes. The route back that the equivalence file names is a new declaration. That declaration is [`configs/gpu_task_qualification.yaml`](../configs/gpu_task_qualification.yaml), which ran the same day ([`docs/GPU_TASK_QUALIFICATION.md`](GPU_TASK_QUALIFICATION.md)). **`host_gpu_det` is `TASK_EQUIVALENT` and `TRAINING_REPRODUCIBLE`**, so it is nameable as a **host-native** placement: every arm a stage compares runs on the host GPU, and its numbers are never compared with laptop numbers (see "The result" below). No stage has named the host yet. Everything else on the host is tests, environment builds, benchmarks and systems validation.
2. **Drive the host only through `python tools/rx/rx.py …`, run from the repository root.** Don't do real work with a raw `ssh gpu "…"`. Windows OpenSSH kills every process a session started when the session ends. rx launches jobs through WMI, which is the only launch method that survives the disconnect.
3. **Move code, not data.** The link is a Tailscale DERP relay: about 0.75 MB/s down and 1.2 MB/s up. The tracked tree (about 11 MB) pushes in seconds. The six datasets' inputs (about 56 GB) would take about 13 hours.
4. **Put no credentials on the host.** It is a shared lab machine. That means no Modal, Hugging Face or cloud tokens, and no keys. Never read `~/.ssh/id_ed25519`.
5. **System and security settings belong to the user.** That includes WSL configuration, sudoers, `.wslconfig`, the firewall, services, Tailscale and Cloudflare WARP. Write the script and the exact command, and the user runs it. `tools/rx/wsl_setup.ps1` is the precedent: an earlier attempt to run it for the user was denied as unauthorized persistence.

## The host

| | |
|---|---|
| ssh alias | `gpu` (in `~/.ssh/config`), Tailscale `100.118.95.8`, account `student2` (an administrator) |
| machine | `DESKTOP-SLQMEQH`, a shared lab workstation; other accounts use it |
| OS | Windows 11 (10.0.26200); the default ssh shell is `cmd.exe` |
| CPU | Intel i9-14900, 32 threads, AVX2 |
| memory | 127.7 GB |
| GPU | NVIDIA RTX 4500 Ada Generation, 24 GB, driver 596.71, about 21 TFLOP/s fp32 matmul measured in a job |
| disk | about 590 GB free on `C:` (2026-09-29) |
| Python | NuGet CPython 3.13.5 at `C:\Users\Student2\rx\python\3.13.5\python.exe`; the Store Python cannot run over ssh |
| rx agent | `C:\Users\Student2\rx\agent\rx_agent-f7319f177f96.py` (replaced automatically when `rx.py` has a newer one) |
| environments | `mpr-cpu@31803e6457ab` (torch 2.8.0+cpu) and `mpr-cu128@62fc45e9e1ba` (torch 2.8.0+cu128); both pin the laptop's closure (numpy 2.3.2, scipy 1.16.1, PyG 2.6.1, …) |
| scheduler capacity | 32 CPUs minus 2 reserved, 127.7 GB minus 8 reserved, GPU 0 |
| WSL | Ubuntu-24.04 (WSL 2), configured by the user on 2026-09-29: user `student2` with passwordless sudo, systemd, a 96 GB and 32-CPU VM, 32 GB swap. It shows as `Stopped` when idle and starts when a `--shell wsl` job needs it |
| clock | about 33 s ahead of the laptop; only the displayed times are affected |

The laptop is the reference machine for every filed number: Intel Family 6 Model 186, 12 logical CPUs, AVX2, 15.7 GB of memory, Python 3.13.5, torch 2.8.0+cpu with MKL 2025.2, numpy 2.3.2, PyG 2.6.1.

### Network

Both from campus and from home, the only path so far is Tailscale's DERP relay. Figures measured on 2026-09-29, on home Wi-Fi where IPv4 UDP never reached STUN:
- about 0.75 MB/s down, whatever the stream count;
- about 1.2 MB/s up on 4 streams.

rx's defaults (`streams = 4`, zlib level 6) were measured as the best choice on this path, so don't tune them again unless a direct path appears.

To check for a direct path, run `tailscale ping 100.118.95.8` ("via DERP" means relayed) and `python tools/rx/rx.py doctor --speed 32`. `tools/rx/README.md` (section "The network") has the campus measurements and the LAN route to add if the laptop is ever on the wired lab network.

## Layout on the host

Everything rx owns lives under `C:\Users\Student2\rx`. Sizes are as of 2026-09-29.

```
C:\Users\Student2\rx\
  host.json, sched.lock   scheduler capacity and its host-wide lock
  agent\                  rx_agent-<sha>.py (the only rx code on the host)
  python\3.13.5\          standalone CPython (NuGet), the base of every env
  envs\                   mpr-cpu@31803e6457ab\, mpr-cu128@62fc45e9e1ba\ (+ .json pointers), 11.2 GB
  projects\mpr\
    ws\                   the workspace: pushed files only; no .git, no outputs\ until a job writes them
    jobs\<id>\            per job: spec.json (with any --var), status.json, heartbeat.json, output.log, outputs.json
    index-ws.json         what the workspace holds (sha per file)
  queue\, active\         jobs waiting / running (empty when idle)
  logs\, downloads\, setup\
  wsl\                    the Ubuntu-24.04 virtual disk (1.5 GB)
```

A job runs **in place** in the workspace (`ws`), with its working directory there. Its outputs are the files under the `outputs` globs that changed while it ran. Two consequences:
- A push made while a job runs changes the files under that job.
- Two sessions pushing different trees to the same `ws` can mix code.

Whenever your working tree differs from what another session may be running, use your own workspace, and pass the same `--ws` to every command: `--ws <name>` on `push`, `run` and `fetch`.

## Files in this repository and on the laptop

| path | what it is |
|---|---|
| `tools/rx/rx.py` | the CLI (standard library only; Python 3.11+): setup, doctor, push, run, ls, status, logs, wait, fetch, cancel, rerun, env, exec, capacity, gc, monitor, dashboard, watch |
| `tools/rx/rx_agent.py` | the host agent: answers one request per ssh session and supervises detached jobs (WMI launch, Job Object, heartbeat, FIFO scheduler with backfill and project priority) |
| `tools/rx/README.md` | the full manual: configuration, transport, transfers, jobs, environments, commands, failure modes, network, validation, WSL, security, porting |
| `tools/rx/test_rx.py` | `python -m pytest tools/rx/test_rx.py -q`, including an end-to-end run on a local pseudo-host |
| `tools/rx/example.rx.toml` | commented template for another project |
| `tools/rx/wsl_setup.ps1` | the one-time WSL configuration. **The user runs it** (see the README's "WSL on the host"); rx never runs it |
| `tools/rx/rx`, `tools/rx/rx.cmd` | thin wrappers for `python tools/rx/rx.py` |
| `rx.toml` | this project's rx configuration: push = git-tracked files (minus `legacy/**`), `max_mb = 512`; run defaults = env `mpr-cpu`, 8 CPUs, 16 GB, outputs `outputs/**` and `logs/**`, `PYTHONHASHSEED=0`; profile `gpu` = `mpr-cu128`, 1 GPU, 8 CPUs, 32 GB; profile `big` = 24 CPUs, 96 GB; both envs pinned in full |
| `.rx/` (git-ignored) | `jobs.json` (jobs launched from this checkout), `hashcache.json`, `receipts.json` (what `rx fetch` wrote; rx never overwrites a local file it didn't write) |
| `~/.rx/hosts/gpu.json` | written by `rx setup gpu`: route `gpu`, the agent path and sha, the Python path, 4 streams, zlib 6, Git's `ssh.exe`. It holds no secret |
| `~/.rx/projects.json` | projects seen by `rx monitor` and `rx watch` |
| `configs/cpu_gpu_equivalence.yaml` | the CPU-GPU equivalence test (below) and its run record `run_record_cpu_gpu_equivalence_2026_09_29` |
| `docs/CPU_GPU_EQUIVALENCE.md` | the test's result: verdicts, failing cells, determinism, clocks |
| `configs/gpu_task_qualification.yaml` | the task-level GPU qualification, its `host_native_protocol`, and its run record `run_record_gpu_task_qualification_2026_09_29` |
| `docs/GPU_TASK_QUALIFICATION.md` | its result: task-level verdicts, training repeat, the host-CPU and TF32 readings beside them |
| `scripts/gpu_task_qualification.py` | its stages (bundle, arm, arms, read, doc, file); imports the equivalence script unchanged |
| `configs/universal_v2.yaml#later_stages.compute_placement` | where the placement rule comes from |
| this file | the handover |

## The rules for scientific work

These come from `configs/universal_v2.yaml#later_stages.compute_placement` and from the placement lines of the UMLP-D0.1, v2.1 and v2.2A declarations. Amendment 7 of `universal_v2.yaml` is `local_only`.

**The test.** [`configs/cpu_gpu_equivalence.yaml`](../configs/cpu_gpu_equivalence.yaml) is the test those files require. In short:
- **Arms.**
  - Reference: the laptop CPU at 8 threads. Floor: the laptop at 4 threads.
  - Candidates: `host_cpu_t8` (`mpr-cpu`) and two GPU modes in `mpr-cu128`, both float32 with TF32 off. `host_gpu_det` requests deterministic algorithms; `host_gpu_default` does not.
- **Models.** The six stage-2 `u_gnn_v2_ef` checkpoints and the pilot `u_mlp_v2_mix` twins s0–s2, each pinned by sha256.
- **Probe.** The first batches of the frozen stage-2 seed-0 draw, packed on the laptop into a bundle, which is the only data that goes to the host. 16 batches were declared; the 1.0 GB size cap cut them to 11 (176 fit-carve queries across all six datasets, 0.97 GB).
- **Tiers.** T1 compares forward per-query centred scores at atol/rtol 1e-5 (inherited from `compute_placement`). T2 compares eval-mode gradients: loss within 1e-5 relative, gradients within 1e-4 relative. T3 (trajectory) and T4 (clock) are descriptive only.
- **Verdicts, per candidate.** `EQUIVALENT_BIT_IDENTICAL`, `EQUIVALENT_WITHIN_TOLERANCE`, `NOT_EQUIVALENT`, or `INCONCLUSIVE_FLOOR`.

**What a pass allows.** A pass only makes a placement *nameable* in a later dated authorization block of a declared stage. That block must:
- quote the tested settings verbatim (host, env name@hash, torch and CUDA versions, the determinism mode that passed, TF32 off, float32, 8 threads for host CPU work);
- run the file's `mirror_verification` before the stage reads any data on the host.

A fail keeps the placement barred. No tolerance is relaxed afterwards.

**The result (2026-09-29).**

| candidate | verdict | detail |
|---|---|---|
| `host_cpu_t8` | `EQUIVALENT_WITHIN_TOLERANCE` | forward bit-identical to the laptop on every cell; gradients within 1.8e-07 relative |
| `host_gpu_det` | `NOT_EQUIVALENT` | 2 forward cells over tolerance by at most 1.28×, both in twin s1 (one metaqa draw, one webqsp draw), identical in the repeat; every gradient cell holds |
| `host_gpu_default` | `NOT_EQUIVALENT` | 245 cells, over tolerance by up to 13×; not reproducible run to run |

No ranking decision differed on any arm: 0 of 176 draws changed any metric. That is reported beside the verdict and does not change it. On the clock, a GNN forward+backward batch takes 5.8 s on the laptop, 3.1 s on the host CPU and 0.36 s on the GPU. The declared routes back for the GPU are a new arm of the same test (for example float64) or a new declaration.

**The task-level qualification (2026-09-29).** That new declaration is [`configs/gpu_task_qualification.yaml`](../configs/gpu_task_qualification.yaml). It asks whether the GPU makes the laptop's *retrieval decisions*, reproducibly. The score-level test above stays as it was, and nothing in it was relaxed.
- **Probe.** 12 fresh stage-2 batches (11–22), 192 fit-carve draws across the six datasets, none of them a draw of the equivalence probe.
- **Criteria**, filed before the bundle existed. Top-1 and top-5 sets must match (a swap is excused only between near-tied candidates: a gap under twice the 1e-5 tolerance). R@5, FullCov@5 and Hit@1 must be equal. The repeat must be bit-identical. A short training run (T5) repeated twice must match exactly: every loss, the final weights' sha256 and the final scores.

| arm | result |
|---|---|
| `host_gpu_det` | **`TASK_EQUIVALENT`**: 2 score cells over tolerance (both in twin s1, again; ratios 1.36 and 1.22), 0 top-1 or top-5 changes, 0 metric changes, repeat bit-identical |
| `host_gpu_det` training | **`TRAINING_REPRODUCIBLE`**: losses, weights sha256 and final scores bit-identical across two fresh processes |
| `host_cpu_t8` (beside) | bit-identical to the laptop on every raw score |
| `host_gpu_tf32` (diagnosis) | 1,152 of 1,152 cells over tolerance (up to 714×) and one top-5 change: TF32 stays off |

**What it opens.** `host_gpu_det`, with exactly the tested settings, is nameable in two ways:
- as an **evaluation** placement for frozen checkpoints at the task level, with both placements named beside every number;
- as the placement of a **host-native** stage, under the file's `host_native_protocol`:
  - every compared arm is fitted and evaluated on the host GPU, and comparators filed on the laptop are rerun there as new draws;
  - inputs are byte-verified by `mirror_verification`;
  - every arm runs from one commit, and every checkpoint is pinned by sha256;
  - one arm is run twice to check the repeat;
  - no host-GPU number is compared with a laptop number.

A GPU fit is still a new draw.

**Placement rules** (filed in the declaration; they bind every stage placed off the laptop):
- Arms whose numbers a stage compares are fitted and evaluated on **one placement**, so a device is never a hidden difference between a message-passing arm and a non-message-passing arm.
- **A GPU fit is a new draw.** Its dropout masks come from the device's generator. It never replaces, pairs with, or averages with the laptop fit of the same seed number. Host CPU fits are treated the same way unless a replica check is declared and passes.
- **Filed arrays and checkpoints are never regenerated elsewhere** to confirm or replace them. That covers M2–M3B, universal-v2 and UMLP.
- **Every record names its placement:** host, env name@hash, device, driver, torch and CUDA versions, determinism and TF32 flags, and thread count. Weights are saved as CPU tensors and pinned by `state_sha256`.
- **Only settings that passed are used.** Batches are packed on the host CPU by the frozen code. The laptop CPU stays the reference.

**Always, regardless of placement:**
- M3B code is byte-pinned. New device plumbing goes into a new module as copies under new names (the declaration's `device_path`); `m3b_train.py` is never edited.
- Test splits are never read, and `V2_HELD_CONFIRMATION` is never read.
- Research discipline:
  - one file per phase;
  - stop after each declaration;
  - commit the rule before the number;
  - systems convenience never edits science.
- The CRAG package (`C:/Users/Swastik/Desktop/CRAG`) is **read-only**. Never write, move, rename or re-hash anything under it, and never copy it into this repository. A host mirror is written by reading it and lives outside the rx workspace. It reaches the frozen code only as an in-memory substitution of `substrate.package_root`; no config file is edited.
- Modal is a separate route: `scripts/spawn_modal_jobs.py` only, never `modal run --detach`. The equivalence test does not cover it; Modal would need its own arm.

## Security and shared-machine etiquette

- **ssh.** rx uses your existing ssh key and `~/.ssh/config`. Never read, print or copy the private key (`~/.ssh/id_ed25519`).
- **Tokens.** Never put Modal, Hugging Face or cloud tokens on the host. `--var K=V` values are written into the job's `spec.json` on the host disk, so `--var` is not for secrets either. Keep any step that needs a token on the laptop.
- **System settings.** Claude does not change system or security settings on either machine: WSL root configuration, sudoers, `.wslconfig`, firewall, services, Windows settings, Tailscale settings. It prepares a script and the exact command, and the user runs it.
- **Cloudflare WARP.** Never turn it on without the user's explicit permission.
- **Data.** Research data goes to no third-party service without consent. Pushing data to the host is a declared step of a declared stage, never a convenience.
- **Other users.** Other accounts use the machine, and the rx scheduler only knows rx jobs. Before heavy GPU work, check `rx doctor` for GPU utilisation and memory. Never kill a process you did not start. Leave the capacity reserve (2 CPUs, 8 GB) as it is.
- **Other Claude sessions.** `rx ls` shows jobs launched from every checkout of this project. Never cancel, rerun or garbage-collect a job you did not launch unless the user asks. Use your own `--ws` when your tree differs.
- **Other projects share the scheduler.** CRAG and Jigsaw sessions on the laptop run this repository's `tools/rx/rx.py` by path, so an agent change here reaches their next job. Jigsaw's `scripts/canonical/launch/lowprio.py` mirrors the scheduler arithmetic and yields to other projects' queued jobs. Run `python -m pytest tools/rx/test_rx.py` before any agent change, and exercise a changed launch path on the host with a one-line job before others reach it.
- **Priority (set 2026-09-29, at the user's request that mpr have priority wherever possible).** `host.json` holds `priority: {mpr: 10}`, and `rx monitor` prints it. A queued mpr job goes ahead of every other project's queued job. While it cannot fit, other projects start only in what is left after its request is set aside. Jobs of every other project run at below-normal OS priority, so they use the idle machine in full and give way under contention. Nothing running is ever stopped: the rule acts at admission. Supervisors started by an older agent keep the old rule until their jobs end, and WSL payloads are outside the priority class. Change it only on the user's word: `rx capacity --priority mpr=10` sets it and `--priority mpr=0` removes it.

## How to use it

Run everything from the repository root on the laptop. `rx` below means `python tools/rx/rx.py`. `JOB` accepts a full id, any unique substring, or `last`.

**Health check (start every session with it):**
```bash
python tools/rx/rx.py doctor
```
A healthy host reports:
- the route `ok` in about 2 s, with clock skew about +33 s;
- the agent `f7319f177f96`;
- CPU, memory and disk figures;
- GPU 0 idle, around 0.5 of 24 GB used;
- capacity `32 cpus - 2 reserved, 127.7 GB - 8 reserved, gpus [0]`;
- both envs;
- WSL `Stopped`.

If the route fails, the host or Tailscale is down; see "Failure modes" in the README. Right after an agent update, one line may say `(stale -- will be replaced)`. That is a display quirk.

**Jobs:**
```bash
python tools/rx/rx.py run -n my-cpu-job -- python scripts/some_script.py --flag
```
- `run` pushes the tracked tree first, then launches and prints the job id at once. Defaults: env `mpr-cpu`, 8 CPUs, 16 GB, `PYTHONHASHSEED=0`, with `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` set to the CPU count.
- `-p gpu` gives `mpr-cu128` with 1 GPU, 8 CPUs and 32 GB; `CUDA_VISIBLE_DEVICES` names the assigned GPU.
- `-p big` gives 24 CPUs and 96 GB.
- Override resources with `--cpus`, `--mem`, `--mem-hard` (a kill limit), `--gpu`/`--gpus`, `--timeout` and `--env`.
- `--shell {exec,cmd,powershell,bash,wsl}` picks the shell. `--shell wsl` runs in Ubuntu with the workspace at `/mnt/c/Users/Student2/rx/projects/mpr/ws`. Native Windows is the default and the better choice unless a tool exists only on Linux; `/mnt/c` is slow for many small files.
- `--inputs GLOB` adds files beyond the tracked tree (for example a bundle under `outputs/`). `--outputs GLOB` changes what counts as the job's outputs.
- `--var K=V` sets a job variable. It is not for secrets.
- `--no-push` launches without pushing.
- A CPU job gets `CUDA_VISIBLE_DEVICES=-1` and cannot touch the GPU, in Windows and in WSL.

**Watching and collecting:**
```bash
python tools/rx/rx.py ls
python tools/rx/rx.py status JOB
python tools/rx/rx.py logs JOB --tail 20
python tools/rx/rx.py wait JOB --fetch
python tools/rx/rx.py fetch JOB
```
- `logs -f` streams output and reconnects by itself.
- `wait` survives outages.
- `fetch` brings the job's outputs back to the same relative paths. A local file it did not write is a conflict (exit code 2); resolve it with `--into DIR` or `--overwrite`.
- `fetch --glob 'outputs/x/**'` takes files by pattern, even while the job runs.

**Pushing without running:**
```bash
python tools/rx/rx.py push --dry-run
python tools/rx/rx.py push --inputs "outputs/some/bundle/**"
```
- A push that would send more than 512 MB is refused; `--force` overrides it. Use `--force` only for a transfer a declaration names; that guard is how bulk data stays off the link by accident.
- An interrupted push resumes at the byte.

**Environments:**
```bash
python tools/rx/rx.py env ls
python tools/rx/rx.py env build mpr-cpu -f
```
- An environment is addressed by the hash of its `[envs.NAME]` table. Editing that table builds a new directory and marks the old one `STALE`.
- A CUDA env whose `verify` touches the GPU needs `gpus = 1` in its table.
- **Don't edit the `[envs.mpr-cpu]` or `[envs.mpr-cu128]` tables.** The equivalence test names `mpr-cpu@31803e6457ab` and `mpr-cu128@62fc45e9e1ba`, and a verdict holds only for those hashes. A rebuild after an edit supersedes the tested directory, and `rx gc` then deletes it host-wide after a day. A different environment is a new env name and, for science, a new arm of the equivalence test.

**Other commands:**
- `rx exec -- CMD` runs a short command now. It dies with the connection, so it is never for real work. Add `--no-ws` to run in the rx home and `--shell powershell` for PowerShell.
- `rx cancel JOB` stops a job's whole process tree. `rx rerun JOB` launches the same spec again.
- `rx capacity` shows the scheduler's shares.
- `rx gc --dry-run` shows what would be pruned. What `gc` removes:
  - finished job directories of this project older than 14 days, always keeping the newest 30 (`--all` covers every project);
  - stale partial files in this project's workspaces;
  - **host-wide**, every env directory that no env pointer names any more and that is older than a day.
- `rx monitor` is a terminal view of hosts and jobs. `rx dashboard --open` is the same in a browser on `127.0.0.1:8765`.
- `rx watch` fetches jobs launched with `--fetch` whenever the host is reachable.

**Driving it from a Claude session:**
- `rx run` returns immediately. Wait for a long job with `rx wait JOB --fetch` as a background Bash command. The Bash tool times out after 10 minutes, and a background command re-invokes you when it exits. Don't poll in a sleep loop.
- Name jobs with `-n`.
- Long jobs should write checkpoints under `outputs/`: a host reboot marks running jobs `lost`, and `rx rerun` can resume from a checkpoint.
- Write any Windows path or backslash through the Write tool, not a Bash heredoc. The Bash tool turns `\t` and `\r` in heredocs into TAB and CR; this already corrupted a README line once.
- Write a PowerShell script meant for the host to a file under `tools/` or `outputs/`, push it, and run it with `--shell powershell` (or `-File`). Don't quote it inline through `cmd.exe`.

## Recipes

**The repository's tests on the host:**
```bash
python tools/rx/rx.py run -n repo-tests -- python -m pytest -q -p no:cacheprovider -rs --continue-on-collection-errors
```
- On 2026-09-29: 6,554 passed in about 2 minutes (8,759 pass locally in about 6).
- The job ends with rc=1, as expected. Every test that fails there needs something rx deliberately leaves behind:
  - `outputs/`, which is never pushed;
  - the git history, since the workspace holds files only;
  - the Modal SDK, which is not installed on the host (15 modules import `modal`, hence `--continue-on-collection-errors`).
- A test that passes locally and fails there for any other reason is a real finding.

**GPU sanity check:**
```bash
python tools/rx/rx.py run -p gpu -n cuda-check -w -- python -c "import torch; x = torch.randn(4096, 4096, device='cuda'); print(torch.__version__, torch.cuda.get_device_name(0), float((x @ x).sum()))"
```

**A job whose outputs come back:** write results under `outputs/<something>/` in the script, then run `rx run -n NAME --fetch -w -- python scripts/x.py`, or run `rx wait NAME --fetch` later.

**Archiving host outputs to the hub, and restoring them onto a new host** (`scripts/host_archive_hf.py`). No credential goes to the host:
- The laptop holds the token and asks the hub for each object's signed upload links.
- The host streams the bytes to those links. It completes an object only when the bytes it sent hash to the planned sha256.
- The laptop then commits the objects at `ws/<rel>` in a private dataset repo, re-reads the tree and checks every file's size and sha256.

Archive one tag (a host job of 1 CPU; one tag per directory keeps each wave inside the links' 10 h life):
```bash
python scripts/host_archive_hf.py drive --repo OWNER/mpr-host-archive --tag NAME --token-name TOKEN --include "outputs/DIR/**" --cpus 1 --mem 0.5
```
Restore it. `--dest .` writes into the workspace; a file already there with other bytes is left alone unless `--overwrite` is given:
```bash
python scripts/host_archive_hf.py drive-restore --repo OWNER/mpr-host-archive --tag NAME --token-name TOKEN --dest .
```
- `status --repo OWNER/mpr-host-archive` checks every manifest in a repo against its tree.
- Upload links live 10 h and download links about 1 h. Both are deleted once used.
- Validated on 2026-10-03 (`archive-t0`, `restore-t0`): three files went up and came back, each matching by sha256.

## Quirks that cost time before

- **Detaching.** Windows OpenSSH puts a session's processes in a Job Object that is killed on disconnect. `start /b`, `DETACHED_PROCESS`, `CREATE_BREAKAWAY_FROM_JOB` and `Start-Process` all die with the session. Only WMI `Win32_Process.Create` escapes it, and rx uses that. rx's WSL jobs go through the same supervisor.
- **An empty `CUDA_VISIBLE_DEVICES`.** Windows CUDA reads `""` as unset and exposes every GPU, while torch counts 0 devices, so an allocation still succeeds. rx exports `-1` for CPU jobs (fixed in `d143ea5`).
- **No connection multiplexing.** Windows OpenSSH resets `ControlMaster` sessions, so each rx request pays a full handshake of about 1–2 s. rx batches requests to keep that cheap.
- **The workspace is files only.** It has no `.git` and no `outputs/` until a job writes one. Anything that reads git history or filed artifacts fails there by design.
- **`nproc` inside a job** reports the job's CPU count, because it honours `OMP_NUM_THREADS`. That is intended.
- **Clock skew** of about +33 s shifts displayed times only; rx computes durations from host timestamps.
- **Line endings.** The working tree mixes CRLF and LF. Every file pin in this repository is sha256 over the bytes with CRLF folded to LF.
- **Transfers.** Numpy arrays, parquet and checkpoints are sent raw after four chunks that don't compress. Expect DERP rates on them.
- **Host reboot.** Running jobs become `lost` (their heartbeat is older than the boot). Nothing survives a reboot, so checkpoint.

## What was validated on 2026-09-29

| job | what it showed |
|---|---|
| `260929-120519-detach-test-c5d0` | a 90 s job kept running after its launch session closed and after a `logs -f` session was killed; rc=0, all 90 ticks, and `wait --fetch` brought them back |
| (push test) | a 12 MB push killed half-way re-sent only the missing 6.0 MB; the sha256 matched |
| `260929-120733-env-mpr-cpu-d1b1` | `mpr-cpu@31803e6457ab` built and verified |
| `260929-120734-env-mpr-cu128-f53a` | the first `mpr-cu128` build failed in `verify` with `Invalid device id`: it ran as a CPU job with an empty `CUDA_VISIBLE_DEVICES`. That led to `-1` for CPU jobs and `gpus = 1` for env builds (`d143ea5`) |
| `260929-121829-env-mpr-cu128-5d97` | `mpr-cu128@62fc45e9e1ba` built; `verify` ran a CUDA matmul on the RTX 4500 Ada |
| `260929-122146-cuda-gpu-job-ff78` | in a `-p gpu` job: fp32 matmul at about 21 TFLOP/s; `torch_scatter` and PyG `GATConv` ran on `cuda:0` |
| `260929-122222-cuda-cpu-job-c9a2` | a CPU job in the same env sees no GPU, and allocation is refused |
| `260929-122314-repo-tests-93fc` | the repository's tests: 6,554 passed; rc=1 for the expected reasons above |
| `260929-123549-wsl-cpu-90e7`, `260929-123629-wsl-gpu-cea0` | `--shell wsl` jobs run as `student2` in the `/mnt/c` workspace with the `RX_*` and thread variables forwarded; CUDA in WSL finds 1 device in a `--gpu` job and none in a CPU job |

## The data problem

Running a real stage on the host means mirroring its inputs. The sizes below cover dense-doc shards, CSR stores, and the v2 fit and select caches:

| dataset | size (GB) |
|---|---|
| squad | 0.25 |
| musique | 5.5 |
| metaqa | 6.0 |
| webqsp | 9.1 |
| hotpotqa | 16.1 |
| 2wiki | 18.8 |
| **total** | **about 56** |

The total is roughly 13 hours of upload at the rate measured on 2026-09-29.

Plan a mirror as a one-off, resumable `rx push --inputs … --force` of exactly the files a declared stage reads. Put it outside the repository, since the CRAG package must never be copied into it. The stage's `mirror_verification` must pass before it reads anything:
- every mirrored input matches the laptop copy by sha256;
- the served-freeze check of `scripts/m3b_compile.open_package` passes on the host;
- the stage's first 16 batches, packed on the host, are `torch.equal` to the laptop's.

A direct Tailscale path, or the laptop on the wired lab network, would change this arithmetic by 10–100×. Measure before planning.

## Where things stand and what comes next

- The host is set up and validated (`8b6d258`, `d143ea5`, `e22dee9`). It was reachable and idle on 2026-09-29.
- The CPU-GPU equivalence test **ran on 2026-09-29** (the user's go-ahead is quoted in its run record). Host CPU passed; host GPU failed in both modes (see above). In practice it took about 17 minutes on the laptop, about 5 minutes of host runtime after about 20 minutes in the shared queue, and 11 minutes of upload for the 0.97 GB bundle.
- A stage that wants the host CPU files a dated authorization block that quotes the passing settings (`DESKTOP-SLQMEQH`, `mpr-cpu@31803e6457ab`, torch 2.8.0+cpu, Windows native, 8 threads per process) and runs mirror verification first. Nothing opens by itself, and a host CPU fit is a new draw, never a laptop seed's replica.
- The task-level qualification **ran on 2026-09-29**: `host_gpu_det` is `TASK_EQUIVALENT` and `TRAINING_REPRODUCIBLE`. The host GPU is nameable for host-native stages (above). In practice it took 12 minutes of upload for the 0.99 GB bundle, 44 s for the host CPU arm, about 2 minutes queued behind another project's GPU job, and 35 s for the three GPU arms.
- mpr has admission priority 10 in the shared scheduler (`f4576f1`): an mpr job goes to the head of the queue, but running jobs are never evicted. Submit to the queue and don't hold jobs back.
