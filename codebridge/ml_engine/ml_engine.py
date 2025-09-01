from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

MODEL_DIR = "yit314/codet5p-220m-merged-ckpt150" # uploaded to HF

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device=="cuda" else torch.float32
tok   = AutoTokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=dtype).to(device)
model.eval(); model.config.use_cache=True

import torch, time
@torch.inference_mode()
def generate_code(task, max_new_tokens=256):
    prompt = f"Generate Python code:\nTask: {task}\nCode:"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         num_beams=4, no_repeat_ngram_size=3, repetition_penalty=1.05)
    return tok.decode(out[0], skip_special_tokens=True)

# print(generate_code("Write a function two_sum(nums, target) that returns indices of two numbers adding up to target."))


def main():
    # interactive
    print("""
Please input an instruction in English
for example: write a function that adds two numbers in Python.
(type quit or exit to terminate).""")
    while True:
        try:
            task = input("\nTask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break
        if task == "" or task.lower() in {"quit", "exit"}:
            print("Bye."); break
        t0 = time.time()
        code = generate_code(task)
        print(f"\n# Time: {time.time()-t0:.2f}s\n")
        print(code)

if __name__ == "__main__":
    main()