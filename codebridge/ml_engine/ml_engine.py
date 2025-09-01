import sys, time, torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# my ckpt150 model
MODEL_DIR = "codebridge/models/codet5p_220m_merged_ckpt150"

MAX_NEW_TOKENS = 256
INP_MAX_LEN = 256

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(model_dir: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    model.config.use_cache = True
    return tok, model

@torch.inference_mode()
def generate_code(tok, model, device: str, task: str) -> str:
    prompt = f"Generate Python code:\nTask: {task}\nCode:"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=INP_MAX_LEN).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=1,
        no_repeat_ngram_size=3,
        repetition_penalty=1.05,
    )
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    device = pick_device()
    print(f"[device] {device}")
    tok, model = load_model(MODEL_DIR, device)

    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:]).strip()
        t0 = time.time()
        code = generate_code(tok, model, device, task)
        print(f"\n# Time: {time.time()-t0:.2f}s\n")
        print(code)
        return

    print("Loaded merged model. Type quit/exit to terminate")
    while True:
        try:
            task = input("\nTask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break
        if task == "" or task.lower() in {"quit", "exit"}:
            print("Bye."); break
        t0 = time.time()
        code = generate_code(tok, model, device, task)
        print(f"\n# Time: {time.time()-t0:.2f}s\n")
        print(code)

if __name__ == "__main__":
    main()