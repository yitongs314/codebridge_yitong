# pip install -U evaluate sacrebleu
import argparse, re, csv
from pathlib import Path
import torch, evaluate
from datasets import load_from_disk
from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import PeftModel
from difflib import SequenceMatcher

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True, help="base model: Ex. Salesforce/codet5p-220m-py")
    ap.add_argument("--lora_ckpt", required=True, help="LoRA checkpoint")
    ap.add_argument("--data_dir",  required=True, help="tokenized data）")
    ap.add_argument("--split", default="test", choices=["train","validation","test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="only evaluate first N")
    ap.add_argument("--out_csv", default="eval_outputs.csv")
    return ap.parse_args()

def pick_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def normalize_ws(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{2,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s

def normalize_strict(s: str) -> str:
    return re.sub(r"\s+", "", s.strip())


def ex_to_text_code(ex, tok):
    if "text" in ex and "code" in ex:
        return ex["text"], ex["code"]

    inp_ids = [i for i in ex["input_ids"] if i != tok.pad_token_id]
    prompt_str = tok.decode(inp_ids, skip_special_tokens=True)
    m = re.search(r"Task:\s*(.*?)\s*Code:\s*$", prompt_str, flags=re.S)
    text = m.group(1).strip() if m else prompt_str.strip()

    labels = ex["labels"]
    labels = [tok.pad_token_id if x == -100 else x for x in labels]
    ref_code = tok.decode(labels, skip_special_tokens=True).strip()
    return text, ref_code



def main():
    args = get_args()
    device = pick_device()

    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    base = T5ForConditionalGeneration.from_pretrained(args.base_ckpt)
    model = PeftModel.from_pretrained(base, args.lora_ckpt)
    model.to(device)
    model.eval()
    model.config.use_cache = True

    ds = load_from_disk(args.data_dir)[args.split]

    if args.limit and args.limit < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.limit))


    bleu = evaluate.load("sacrebleu")
    preds, refs, rows = [], [], []


    for ex in ds:
        text, ref_code = ex_to_text_code(ex, tok)
        # text, ref_code = ex["text"], ex["code"]
        prompt = f"Generate Python code:\nTask: {text}\nCode:"
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=384).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, num_beams=1)
        pred = tok.decode(out[0], skip_special_tokens=True)

        preds.append(pred)
        refs.append([ref_code])
        rows.append({"text": text, "pred": pred, "ref": ref_code})

    # indicators
    bleu_score = bleu.compute(predictions=preds, references=refs)["score"]
    em_ws = sum(normalize_ws(p) == normalize_ws(r[0]) for p, r in zip(preds, refs)) / len(preds)
    em_strict = sum(normalize_strict(p) == normalize_strict(r[0]) for p, r in zip(preds, refs)) / len(preds)
    ratios = [SequenceMatcher(None, normalize_ws(p), normalize_ws(r[0])).ratio()
              for p, r in zip(preds, refs)]
    avg_ratio = sum(ratios) / len(ratios)

    # main indicators:
    # bleu_score
    # 空格不敏感字符串全等
    # 空格敏感字符串全等
    # difflib ratio

    print(f"[Split={args.split}] N={len(preds)}")
    print(f"BLEU: {bleu_score:.2f}")
    print(f"ExactMatch (whitespace-insensitive): {em_ws*100:.2f}%")
    print(f"ExactMatch (remove-all-whitespace): {em_strict*100:.2f}%")
    print(f"Avg Similarity (difflib ratio): {avg_ratio*100:.2f}%")

    out_path = Path(args.out_csv)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text","pred","ref"])
        w.writeheader()
        w.writerows(rows)
    print("Saved examples to:", out_path.resolve())

if __name__ == "__main__":
    main()