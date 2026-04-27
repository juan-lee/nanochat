# Ray distributed training on Skyline

This repo can run nanochat training across multiple Skyline GPU nodes with KubeRay. The current validated target is one GB10 GPU per Spark node.

## Required container changes

Use the repo's two images:

- GPU worker image: `Dockerfile`
  - Ubuntu 24.04 base.
  - Bundles the known-good ARM64 `ptxas` at `/usr/local/cuda-13.0/bin/ptxas`.
  - Installs the project venv with `uv sync --extra gpu --extra ray` so Ray workers have nanochat, CUDA PyTorch, Transformers for `base_eval`, Ray, pyarrow, setproctitle, and dashboard support packages.
  - Installs `procps` for process inspection.
  - Marks `.venv/bin/py-spy` setuid root so Ray dashboard traceback and CPU profiling work from the dashboard agent.
- Ray head image: `Dockerfile.ray-head`
  - Python 3.12.3 slim image.
  - Installs `ray[default]==2.52.1`, `pyarrow>=21.0.0`, and `setproctitle>=1.3.7`.
  - Marks `/usr/local/bin/py-spy` setuid root for Ray dashboard traceback and CPU profiling on the head node.

Build/push workflows:

- `.github/workflows/container.yml` builds the GPU worker image for `linux/arm64`.
- `.github/workflows/ray-head-container.yml` builds the Ray head image for `linux/amd64`.

Prefer explicit image tags or commit SHAs in cluster manifests. Avoid `:latest` and `imagePullPolicy: Always` for repeated validation runs so cached images are reused.

## KubeRay cluster shape

The validated cluster has:

- 1 Ray head pod on a CPU/control node.
- 2 Ray worker pods, one per Spark GB10 node.
- Worker resources include `num_gpus=1` and custom resource `accelerator_type:GB10`.
- Ray dashboard/API exposed through the head service on port `8265`.
- Ray GCS on port `6379`.

For local control, port-forward the dashboard/API:

```bash
export KUBECONFIG=/Users/jpang/.openclaw/workspace/.kube/skyline-config
kubectl -n default port-forward svc/nanochat-ray-smoke-pinned-head-svc 8265:8265
```

Then use the Python Ray CLI, not Homebrew's unrelated `ray` debugger:

```bash
export PATH="$HOME/.local/bin:$PATH"
ray --version
ray job list --address http://127.0.0.1:8265
```

## Orchestration driver

`scripts/ray_distributed_nanochat.py` is the distributed driver.

It does the following:

1. Connects to the existing Ray cluster with `ray.init(address="auto")`.
2. Starts one `RankActor` per distributed rank.
3. Places actors on GB10 workers via `@ray.remote(num_gpus=1, resources={"accelerator_type:GB10": 0.01})`.
4. Copies the repo workspace into a per-run directory on each actor, excluding `.venv` and `.git`.
5. Reuses or creates a local venv and runs `uv sync --extra gpu` on each actor.
6. Runs dataset/tokenizer phases on all ranks.
7. Runs distributed phases with `torchrun --nnodes=<num-ranks> --nproc_per_node=1`.
8. Uses rank 0's actor IP as `--master_addr` and a configurable `--master-port` base.
9. Sets runtime env such as `NANOCHAT_BASE_DIR`, `WANDB_RUN=dummy`, `OMP_NUM_THREADS=1`, `NCCL_DEBUG=INFO`, and `NCCL_SOCKET_IFNAME=eth0`.

Example validation submission:

```bash
ray job submit \
  --address http://127.0.0.1:8265 \
  --submission-id ray-ddp-depth12-baseeval-sample-$(date -u +%Y%m%d-%H%M%S) \
  --working-dir . \
  --entrypoint-num-cpus 1 \
  --no-wait \
  -- python scripts/ray_distributed_nanochat.py \
    --run-id ray-ddp-depth12-$(date -u +%Y%m%d-%H%M%S) \
    --depth 12 \
    --target-ratio 8 \
    --device-batch-size 16 \
    --total-shards 170 \
    --master-port 29613 \
    --base-eval sample \
    --base-eval-split-tokens 524288 \
    --skip-chat-sft \
    --skip-chat-eval \
    --skip-report
```


## Durable run state with Longhorn RWX

Long-running validation runs should not rely on pod-local `/tmp` for state that
must survive pod restarts or KubeRay reconciliation. The Skyline manifest defines
a Longhorn RWX PVC named `nanochat-runs-rwx` and mounts it at:

```text
/mnt/nanochat-runs
```

Use local scratch for per-rank workspace/venv/compile cache and the RWX mount for
checkpoints, reports, and logs:

```text
/tmp/<run-id>/workspace                         # local scratch per worker
/mnt/nanochat-runs/<run-id>/cache              # durable NANOCHAT_BASE_DIR
/mnt/nanochat-runs/<run-id>/logs               # durable per-actor logs
```

The driver exposes these path controls:

- `--run-root`: per-rank scratch root; defaults to `/tmp/<run-id>`.
- `--cache-dir`: durable `NANOCHAT_BASE_DIR`; defaults to `<run-root>/cache`.
- `--log-dir`: durable actor log directory; defaults to `<run-root>`.
- `--skip-prep`: skip report reset, dataset/tokenizer prep, and tokenizer eval
  when resuming a run from existing durable state.

Example durable base validation run:

```bash
RUN_ID=ray-ddp-depth12-$(date -u +%Y%m%d-%H%M%S)
ray job submit \
  --address http://127.0.0.1:8265 \
  --submission-id ${RUN_ID}-baseeval-sample \
  --working-dir . \
  --entrypoint-num-cpus 1 \
  --no-wait \
  -- python scripts/ray_distributed_nanochat.py \
    --run-id "$RUN_ID" \
    --run-root "/tmp/$RUN_ID" \
    --cache-dir "/mnt/nanochat-runs/$RUN_ID/cache" \
    --log-dir "/mnt/nanochat-runs/$RUN_ID/logs" \
    --fresh \
    --depth 12 \
    --target-ratio 8 \
    --device-batch-size 16 \
    --total-shards 170 \
    --master-port 29613 \
    --base-eval sample \
    --base-eval-split-tokens 524288 \
    --skip-chat-sft \
    --skip-chat-eval \
    --skip-report
```

Example continuation from the same durable base checkpoint into chat SFT/eval:

```bash
ray job submit \
  --address http://127.0.0.1:8265 \
  --submission-id ${RUN_ID}-chat-resume \
  --working-dir . \
  --entrypoint-num-cpus 1 \
  --no-wait \
  -- python scripts/ray_distributed_nanochat.py \
    --run-id "$RUN_ID" \
    --run-root "/tmp/$RUN_ID" \
    --cache-dir "/mnt/nanochat-runs/$RUN_ID/cache" \
    --log-dir "/mnt/nanochat-runs/$RUN_ID/logs" \
    --depth 12 \
    --target-ratio 8 \
    --device-batch-size 16 \
    --total-shards 170 \
    --master-port 29713 \
    --skip-prep \
    --skip-base-train \
    --skip-base-eval
```

Longhorn RWX is NFS-backed. It is a good fit for durable checkpoints, reports,
and logs. Keep the uploaded Ray working directory, venv, Torch compile cache, and
other hot scratch data on local disk unless profiling shows shared storage is
fast enough.

## Checkpoint locality fix

Distributed nanochat training writes the global model checkpoint from distributed rank 0 only. Optimizer state is rank-local.

That means downstream phases cannot assume every node has these files locally:

- `model_*.pt`
- `meta_*.json`

The driver solves this with `broadcast_model_meta`:

1. Rank 0 tars and base64-encodes only `model_*.pt` and `meta_*.json` from the checkpoint directory.
2. Other actors unpack those files into their local checkpoint directory.
3. `optim_*.pt` is intentionally excluded so rank-local optimizer shards are not overwritten.

This broadcast runs after base training and after chat SFT.

## Eval controls

Full `base_eval` can be slow. The driver exposes passthrough flags:

- `--base-eval`: comma-separated eval modes, default `core,bpb,sample`.
- `--base-eval-max-per-task`: forwarded to `scripts.base_eval --max-per-task`.
- `--base-eval-split-tokens`: forwarded to `scripts.base_eval --split-tokens`.

For fast orchestration validation, use `--base-eval sample`. For a quick BPB check, use `--base-eval bpb --base-eval-split-tokens 524288`.

## Debugging

Useful Ray commands:

```bash
ray job status --address http://127.0.0.1:8265 <submission-id>
ray job logs --address http://127.0.0.1:8265 <submission-id>
ray list workers --address http://127.0.0.1:8265 --format table
ray summary tasks --address http://127.0.0.1:8265
```

Per-actor logs live under the run directory on each worker:

```text
/tmp/<run-id>/actor-<hostname>.log
```

If a stopped or crashed job leaves stale `torchrun` processes, clean them before reusing the same master port, or submit with a different `--master-port`.

Ray dashboard traceback and CPU profiling require `py-spy` plus `SYS_PTRACE` on the Ray pods. The Dockerfiles mark `py-spy` setuid root, and the KubeRay pod specs should add this container security context:

```yaml
securityContext:
  capabilities:
    add:
    - SYS_PTRACE
```

Without `SYS_PTRACE`, dashboard links such as `/worker/traceback` and `/worker/cpu_profile` can fail with `Error: Failed to copy Py_Version symbol` / `Permission denied (os error 13)` even when `py-spy` is installed.

## Validated result

The checkpoint locality fix was validated with Ray job `ray-ddp-depth12-baseeval-sample-20260426-0812` against run directory `/tmp/ray-ddp-depth12-20260426-053201`.

Validation confirmed:

- Rank 0 broadcast `model_001680.pt` and `meta_001680.json` from `/tmp/ray-ddp-depth12-20260426-053201/cache/base_checkpoints/d12`.
- Rank 1 received those files.
- `base_eval` completed on both ranks with `rc: 0`.
