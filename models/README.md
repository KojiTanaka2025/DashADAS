# Model weights (not committed)

Download YOLOPv2 TorchScript weights on the GPU guest:

```bash
./scripts/download-yolopv2.sh
```

Expected path: `models/yolopv2.pt` (~149MB). Mounted read-only into the app container as `/app/models`.
