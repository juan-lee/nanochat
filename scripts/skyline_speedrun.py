import argparse
from textwrap import dedent

DEFAULT_HOST_ROOT = "/var/lib/skyline/nanochat"
DEFAULT_NS = "default"

def env_line(name, value):
    return {"name": name, "value": str(value)}

def main():
    p = argparse.ArgumentParser(description="Emit a Kubernetes Job manifest for nanochat speedrun on Skyline")
    p.add_argument("--job-name", default="nanochat-speedrun")
    p.add_argument("--namespace", default=DEFAULT_NS)
    p.add_argument("--image", default="ghcr.io/juan-lee/nanochat:latest")
    p.add_argument("--image-pull-policy", default="Always")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--cpu", default="16")
    p.add_argument("--memory", default="96Gi")
    p.add_argument("--shared-memory", default="16Gi")
    p.add_argument("--host-root", default=DEFAULT_HOST_ROOT)
    p.add_argument("--node-selector", action="append", default=["skyline.ai/gpu=true"])
    p.add_argument("--toleration-key", default="nvidia.com/gpu")
    p.add_argument("--nproc-per-node", type=int, default=1)
    p.add_argument("--base-depth", type=int, default=12)
    p.add_argument("--base-device-batch-size", type=int, default=2)
    p.add_argument("--base-eval-device-batch-size", type=int, default=2)
    p.add_argument("--sft-device-batch-size", type=int, default=2)
    p.add_argument("--base-target-param-data-ratio", type=float, default=12)
    p.add_argument("--pretrain-total-shards", type=int, default=24)
    p.add_argument("--tokenizer-bootstrap-shards", type=int, default=8)
    p.add_argument("--enable-fp8", default="0")
    p.add_argument("--base-extra-arg", action="append", default=[])
    p.add_argument("--base-eval-extra-arg", action="append", default=[])
    p.add_argument("--sft-extra-arg", action="append", default=[])
    p.add_argument("--chat-eval-extra-arg", action="append", default=[])
    p.add_argument("--tok-train-arg", action="append", default=[])
    p.add_argument("--tok-eval-arg", action="append", default=[])
    args = p.parse_args()

    selector = dict(item.split("=", 1) for item in args.node_selector)
    run_root = f"{args.host_root}/{args.job_name}"
    env = [
        env_line("OMP_NUM_THREADS", "1"),
        env_line("NANOCHAT_BASE_DIR", "/artifacts/cache"),
        env_line("NPROC_PER_NODE", args.nproc_per_node),
        env_line("BASE_DEPTH", args.base_depth),
        env_line("BASE_DEVICE_BATCH_SIZE", args.base_device_batch_size),
        env_line("BASE_EVAL_DEVICE_BATCH_SIZE", args.base_eval_device_batch_size),
        env_line("SFT_DEVICE_BATCH_SIZE", args.sft_device_batch_size),
        env_line("BASE_TARGET_PARAM_DATA_RATIO", args.base_target_param_data_ratio),
        env_line("PRETRAIN_TOTAL_SHARDS", args.pretrain_total_shards),
        env_line("TOKENIZER_BOOTSTRAP_SHARDS", args.tokenizer_bootstrap_shards),
        env_line("ENABLE_FP8", args.enable_fp8),
        env_line("NANOCHAT_DISABLE_COMPILE", "1"),
        env_line("NANOCHAT_DISABLE_FUSED_OPTIM", "1"),
        env_line("NANOCHAT_DISABLE_MUON", "1"),
        env_line("BASE_EXTRA_ARGS", " ".join(args.base_extra_arg)),
        env_line("BASE_EVAL_EXTRA_ARGS", " ".join(args.base_eval_extra_arg)),
        env_line("SFT_EXTRA_ARGS", " ".join(args.sft_extra_arg)),
        env_line("CHAT_EVAL_EXTRA_ARGS", " ".join(args.chat_eval_extra_arg)),
        env_line("TOK_TRAIN_ARGS", " ".join(args.tok_train_arg)),
        env_line("TOK_EVAL_ARGS", " ".join(args.tok_eval_arg)),
    ]

    import yaml
    doc = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": args.job_name, "namespace": args.namespace},
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "runtimeClassName": "nvidia",
                    "nodeSelector": selector,
                    "tolerations": [{
                        "key": args.toleration_key,
                        "operator": "Equal",
                        "value": "true",
                        "effect": "NoSchedule",
                    }],
                    "volumes": [
                        {"name": "artifacts", "hostPath": {"path": run_root, "type": "DirectoryOrCreate"}},
                        {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": args.shared_memory}},
                    ],
                    "containers": [{
                        "name": "speedrun",
                        "image": args.image,
                        "imagePullPolicy": args.image_pull_policy,
                        "workingDir": "/workspace",
                        "env": env,
                        "command": ["bash", "-lc"],
                        "args": [dedent("""
                            set -euo pipefail
                            mkdir -p /artifacts/cache /artifacts/output
                            bash runs/speedrun.sh 2>&1 | tee /artifacts/output/speedrun.log
                        """).strip()],
                        "resources": {
                            "requests": {"cpu": args.cpu, "memory": args.memory, "nvidia.com/gpu": args.gpu_count},
                            "limits": {"cpu": args.cpu, "memory": args.memory, "nvidia.com/gpu": args.gpu_count},
                        },
                        "volumeMounts": [
                            {"name": "artifacts", "mountPath": "/artifacts"},
                            {"name": "dshm", "mountPath": "/dev/shm"},
                        ],
                    }],
                }
            },
        },
    }
    print(yaml.safe_dump(doc, sort_keys=False))

if __name__ == "__main__":
    main()
