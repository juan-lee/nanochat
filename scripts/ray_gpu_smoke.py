"""Smoke-test nanochat's Ray environment on GPU workers.

Run from a Ray head pod/container:

    python -m scripts.ray_gpu_smoke --num-workers 2
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess

import ray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--address", default="auto", help="Ray address, default: auto")
    p.add_argument("--num-workers", type=int, default=1)
    return p.parse_args()


@ray.remote(num_gpus=1)
def gpu_probe(index: int) -> dict:
    import torch

    result = {
        "index": index,
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ray_gpu_ids": ray.get_gpu_ids(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        result["torch_device_name"] = torch.cuda.get_device_name(0)

    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=10,
        )
        result["nvidia_smi_rc"] = p.returncode
        result["nvidia_smi_stdout"] = p.stdout.strip()
        result["nvidia_smi_stderr"] = p.stderr.strip()
    except Exception as e:  # pragma: no cover - diagnostic path
        result["nvidia_smi_error"] = repr(e)
    return result


def main() -> None:
    args = parse_args()
    ray.init(address=args.address)
    print("cluster_resources", ray.cluster_resources())
    print("available_resources", ray.available_resources())
    results = ray.get([gpu_probe.remote(i) for i in range(args.num_workers)])
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
