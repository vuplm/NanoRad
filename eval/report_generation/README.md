Run baseline model:
```bash
python run_baseline_eval.py \
  --ckpt /path/to/checkpoint.ckpt \
  --test_json /path/to/iuxray_test.json \
  --image_root /path/to/iu_images
```

Run advance model:
```bash
python run_advance_eval.py \
  --ckpt /path/to/checkpoint.ckpt \
  --test_json /path/to/iuxray_test.json \
  --image_root /path/to/iu_images
```
