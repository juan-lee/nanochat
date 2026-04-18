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
  --base-extra-arg='--window-pattern=L' \
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
