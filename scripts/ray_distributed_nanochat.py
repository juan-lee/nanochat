#!/usr/bin/env python3
"""Run nanochat phases across KubeRay GPU workers.

This driver uses one Ray actor per GPU worker and launches torchrun across those
actors. Checkpoints are saved by torch distributed rank 0 only, while optimizer
state is sharded per rank. After phases that create a model checkpoint, this
script broadcasts rank 0's model/meta files to the other ranks without touching
their rank-local optimizer shards.
"""

from __future__ import annotations

import argparse
import base64
import os
import shlex
import socket
import subprocess
import tarfile
import time
from pathlib import Path

import ray


@ray.remote(num_gpus=1, resources={"accelerator_type:GB10": 0.01})
class RankActor:
    def __init__(self, rank: int):
        self.rank = rank
        self.hostname = socket.gethostname()
        self.ip = socket.gethostbyname(self.hostname)

    def info(self) -> dict:
        return {"rank": self.rank, "hostname": self.hostname, "ip": self.ip}

    def run(self, phase: str, command: str, cwd: str, env: dict[str, str], log_path: str) -> dict:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        merged_env = os.environ.copy()
        merged_env.update(env)
        merged_env["RANK"] = str(self.rank)
        merged_env["LOCAL_RANK"] = "0"
        merged_env.setdefault("OMP_NUM_THREADS", "1")
        merged_env.setdefault("NCCL_DEBUG", "INFO")
        merged_env.setdefault("NCCL_SOCKET_IFNAME", "eth0")
        merged_env.setdefault("PYTHONUNBUFFERED", "1")

        started = time.time()
        with open(log_path, "a", encoding="utf-8", buffering=1) as log:
            log.write("\n" + "=" * 80 + "\n")
            log.write(f"phase={phase} rank={self.rank} host={self.hostname} ip={self.ip}\n")
            log.write(f"cwd={cwd}\n")
            log.write(f"command={command}\n")
            log.write("=" * 80 + "\n")
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                env=merged_env,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return {
            "phase": phase,
            "rank": self.rank,
            "rc": proc.returncode,
            "hostname": self.hostname,
            "ip": self.ip,
            "elapsed_sec": round(time.time() - started, 3),
            "log": log_path,
        }

    def remove_path(self, path: str) -> dict:
        subprocess.run(f"rm -rf {shlex.quote(path)}", shell=True, check=True)
        return {"rank": self.rank, "removed": path}

    def ensure_workspace(self, run_root: str, cache_dir: str, fresh: bool) -> dict:
        workdir = f"{run_root}/workspace"
        if fresh:
            subprocess.run(f"rm -rf {shlex.quote(workdir)}", shell=True, check=True)
        Path(workdir).mkdir(parents=True, exist_ok=True)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        if not Path(workdir, "pyproject.toml").exists():
            cmd = f"cd /workspace && tar --exclude='./.venv' --exclude='./.git' -cf - . | tar -C {shlex.quote(workdir)} -xf -"
            subprocess.run(cmd, shell=True, check=True)
        return {"rank": self.rank, "workdir": workdir, "cache_dir": cache_dir}

    def tar_model_meta_b64(self, checkpoint_dir: str) -> str:
        # Model/meta are global and written only by distributed rank 0. Optimizer
        # shards are rank-local, so deliberately exclude optim_*.pt.
        path = Path(checkpoint_dir)
        files = sorted(list(path.glob("model_*.pt")) + list(path.glob("meta_*.json")))
        if not files:
            raise FileNotFoundError(f"no model/meta checkpoint files in {checkpoint_dir}")
        tar_path = path.parent / f".{path.name}-model-meta.tar"
        with tarfile.open(tar_path, "w") as tf:
            for file in files:
                tf.add(file, arcname=file.name)
        return base64.b64encode(tar_path.read_bytes()).decode("ascii")

    def untar_model_meta_b64(self, checkpoint_dir: str, payload_b64: str) -> dict:
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        tar_path = path.parent / f".{path.name}-model-meta.recv.tar"
        tar_path.write_bytes(base64.b64decode(payload_b64.encode("ascii")))
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(path)
        files = sorted(p.name for p in path.glob("model_*.pt")) + sorted(p.name for p in path.glob("meta_*.json"))
        return {"rank": self.rank, "checkpoint_dir": checkpoint_dir, "files": files}


def quote_cmd(argv: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in argv)


def torchrun_cmd(module: str, module_args: list[str], rank: int, nnodes: int, master_addr: str, master_port: int) -> str:
    argv = [
        "torchrun",
        f"--nnodes={nnodes}",
        "--nproc_per_node=1",
        f"--node_rank={rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "-m",
        module,
    ]
    if module_args:
        argv.append("--")
        argv.extend(module_args)
    return quote_cmd(argv)


def run_phase(actors, phase: str, command_for_rank, workdirs, env, log_paths):
    refs = [
        actor.run.remote(phase, command_for_rank(i), workdirs[i], env, log_paths[i])
        for i, actor in enumerate(actors)
    ]
    results = ray.get(refs)
    print(f"Phase {phase} results: {results}", flush=True)
    failures = [r for r in results if r["rc"] != 0]
    if failures:
        raise RuntimeError(f"phase {phase} failed: {failures}")
    return results


def broadcast_model_meta(actors, checkpoint_dir: str):
    print(f"Broadcasting model/meta from rank 0: {checkpoint_dir}", flush=True)
    payload = ray.get(actors[0].tar_model_meta_b64.remote(checkpoint_dir))
    results = ray.get([actor.untar_model_meta_b64.remote(checkpoint_dir, payload) for actor in actors[1:]])
    print(f"Broadcast results: {results}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--target-ratio", type=int, default=8)
    parser.add_argument("--device-batch-size", type=int, default=16)
    parser.add_argument("--total-shards", type=int, default=170)
    parser.add_argument("--num-ranks", type=int, default=2)
    parser.add_argument("--master-port", type=int, default=29513)
    parser.add_argument("--run-root", help="Per-rank scratch root for workspace/venv/compile cache. Defaults to /tmp/<run-id>.")
    parser.add_argument("--cache-dir", help="Durable NANOCHAT_BASE_DIR for checkpoints, reports, and nanochat cache. Defaults to <run-root>/cache.")
    parser.add_argument("--log-dir", help="Directory for per-actor logs. Defaults to <run-root>.")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--skip-prep", action="store_true", help="Skip report reset, dataset/tokenizer prep, and tokenizer eval when resuming from an existing run dir.")
    parser.add_argument("--skip-base-train", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--skip-chat-sft", action="store_true")
    parser.add_argument("--skip-chat-eval", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument(
        "--base-eval",
        default="core,bpb,sample",
        help="Comma-separated base_eval modes; use bpb or sample for quick orchestration validation.",
    )
    parser.add_argument(
        "--base-eval-max-per-task",
        type=int,
        default=-1,
        help="Forwarded to scripts.base_eval --max-per-task for quicker CORE validation.",
    )
    parser.add_argument(
        "--base-eval-split-tokens",
        type=int,
        default=40 * 524288,
        help="Forwarded to scripts.base_eval --split-tokens; lower it for quick BPB validation.",
    )
    args = parser.parse_args()

    ray.init(address="auto")
    run_root = args.run_root or f"/tmp/{args.run_id}"
    cache_dir = args.cache_dir or f"{run_root}/cache"
    log_dir = args.log_dir or run_root

    actors = [RankActor.remote(i) for i in range(args.num_ranks)]
    infos = ray.get([a.info.remote() for a in actors])
    print(f"Actors: {infos}", flush=True)
    print(f"Paths: run_root={run_root} cache_dir={cache_dir} log_dir={log_dir}", flush=True)
    master_addr = infos[0]["ip"]

    if args.fresh:
        # cache_dir may be an RWX mount shared by all ranks; clear it once from rank 0.
        print(ray.get(actors[0].remove_path.remote(cache_dir)), flush=True)

    setup = ray.get([a.ensure_workspace.remote(run_root, cache_dir, args.fresh) for a in actors])
    workdirs = [s["workdir"] for s in setup]
    logs = [f"{log_dir}/actor-{info['hostname']}.log" for info in infos]

    env = {
        "NANOCHAT_BASE_DIR": cache_dir,
        "WANDB_RUN": "dummy",
        "OMP_NUM_THREADS": "1",
    }

    # Install dependencies once per rank/workdir. Safe to rerun for continuation jobs.
    run_phase(
        actors,
        "setup",
        lambda i: "bash -lc 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; [ -d .venv ] || uv venv; uv sync --extra gpu'",
        workdirs,
        env,
        logs,
    )

    if not args.skip_prep:
        run_phase(actors, "report_reset", lambda i: "bash -lc 'source .venv/bin/activate && python -m nanochat.report reset'", workdirs, env, logs)
        run_phase(actors, "dataset8", lambda i: "bash -lc 'source .venv/bin/activate && python -m nanochat.dataset -n 8'", workdirs, env, logs)
        run_phase(actors, "dataset_total", lambda i: f"bash -lc 'source .venv/bin/activate && python -m nanochat.dataset -n {args.total_shards}'", workdirs, env, logs)
        run_phase(actors, "tok_train", lambda i: "bash -lc 'source .venv/bin/activate && python -m scripts.tok_train'", workdirs, env, logs)
        run_phase(actors, "tok_eval", lambda i: "bash -lc 'source .venv/bin/activate && python -m scripts.tok_eval'", workdirs, env, logs)

    model_tag = f"d{args.depth}"
    base_ckpt = f"{cache_dir}/base_checkpoints/{model_tag}"
    chatsft_ckpt = f"{cache_dir}/chatsft_checkpoints/{model_tag}"

    if not args.skip_base_train:
        run_phase(
            actors,
            "base_train",
            lambda i: "bash -lc " + shlex.quote(
                "source .venv/bin/activate && "
                + torchrun_cmd(
                    "scripts.base_train",
                    [
                        f"--depth={args.depth}",
                        f"--target-param-data-ratio={args.target_ratio}",
                        f"--device-batch-size={args.device_batch_size}",
                        "--run=dummy",
                    ],
                    i,
                    args.num_ranks,
                    master_addr,
                    args.master_port,
                )
            ),
            workdirs,
            env,
            logs,
        )
    broadcast_model_meta(actors, base_ckpt)

    if not args.skip_base_eval:
        base_eval_args = [
            f"--model-tag={model_tag}",
            f"--device-batch-size={args.device_batch_size}",
            f"--eval={args.base_eval}",
            f"--max-per-task={args.base_eval_max_per_task}",
            f"--split-tokens={args.base_eval_split_tokens}",
        ]
        run_phase(
            actors,
            "base_eval",
            lambda i: "bash -lc " + shlex.quote(
                "source .venv/bin/activate && "
                + torchrun_cmd("scripts.base_eval", base_eval_args, i, args.num_ranks, master_addr, args.master_port + 1)
            ),
            workdirs,
            env,
            logs,
        )

    if not args.skip_chat_sft:
        run_phase(
            actors,
            "identity_data",
            lambda i: f"bash -lc 'source .venv/bin/activate && curl -L -o {shlex.quote(cache_dir)}/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl'",
            workdirs,
            env,
            logs,
        )
        run_phase(
            actors,
            "chat_sft",
            lambda i: "bash -lc " + shlex.quote(
                "source .venv/bin/activate && "
                + torchrun_cmd(
                    "scripts.chat_sft",
                    [f"--model-tag={model_tag}", f"--device-batch-size={args.device_batch_size}", "--run=dummy"],
                    i,
                    args.num_ranks,
                    master_addr,
                    args.master_port + 2,
                )
            ),
            workdirs,
            env,
            logs,
        )
        broadcast_model_meta(actors, chatsft_ckpt)

    if not args.skip_chat_eval:
        run_phase(
            actors,
            "chat_eval",
            lambda i: "bash -lc " + shlex.quote(
                "source .venv/bin/activate && "
                + torchrun_cmd("scripts.chat_eval", ["-i", "sft", f"--model-tag={model_tag}"], i, args.num_ranks, master_addr, args.master_port + 3)
            ),
            workdirs,
            env,
            logs,
        )

    if not args.skip_report:
        # Report/chat_cli only need rank 0 and the final checkpoint.
        run_phase(
            [actors[0]],
            "report_generate",
            lambda i: f"bash -lc 'source .venv/bin/activate && python -m nanochat.report generate && python -m scripts.chat_cli --model-tag {shlex.quote(model_tag)} -p \"What is the capital of France?\"'",
            [workdirs[0]],
            env,
            [logs[0]],
        )

    print(f"Done. run_root={run_root} cache_dir={cache_dir} log_dir={log_dir} logs={logs}", flush=True)


if __name__ == "__main__":
    main()
