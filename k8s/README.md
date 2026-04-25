# Skyline one-off speedrun job

Build and push the image via GitHub Actions, then generate a one-off Job manifest:

```bash
python -m scripts.skyline_speedrun \
  --image ghcr.io/juan-lee/nanochat:latest \
  --job-name nanochat-speedrun-gb10 \
  --gpu-count 1 \
  --nproc-per-node 1 \
  --base-depth 12 \
  --base-device-batch-size 2 \
  --base-eval-device-batch-size 2 \
  --sft-device-batch-size 2 \
  --enable-fp8 0 \
  --pretrain-total-shards 24 \
  --base-extra-arg='--core-metric-every=-1' \
  --base-extra-arg='--sample-every=-1' \
  --base-extra-arg='--eval-every=250' \
  --base-extra-arg='--save-every=-1' \
  --base-eval-extra-arg='--max-per-task=100' \
  --base-eval-extra-arg='--split-tokens=524288' \
  --chat-eval-extra-arg='-a ARC-Easy' \
  > /tmp/nanochat-speedrun-gb10.yaml

kubectl apply -f /tmp/nanochat-speedrun-gb10.yaml
kubectl logs -f job/nanochat-speedrun-gb10
```

Artifacts and logs land on the selected GPU node under:

```text
/var/lib/skyline/nanochat/<job-name>
```

## Last minimal two-Spark run

The manifests from the 2026-04-23/24 minimal two-Spark run are saved under `k8s/skyline/`:

- `nanochat-speedrun-spark-34b0-minimal.yaml`
- `nanochat-speedrun-spark-34ef-minimal.yaml`

These preserve the node-specific hostPath artifact locations and the minimal environment used for that run. The post-run chat smoke test now lives inside `runs/speedrun.sh`, where the uv virtualenv is still active, instead of being appended by the Kubernetes wrapper.
