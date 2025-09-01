### This project loads the merged CodeT5p weights directly from the Hugging Face Hub (no local model files required).

### Model:
HF model ID: yit314/codet5p-220m-merged-ckpt150

### Notes
Device auto-selection: CUDA → MPS (Apple) → CPU.\
Defaults favor speed: num_beams=1, max_new_tokens≈256.\
Run rules_engine/main to access rule-based engine (part 1)\
Run ml_engine/ml_engine to access ml_based engine (part 2)
