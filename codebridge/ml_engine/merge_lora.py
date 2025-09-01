from transformers import AutoTokenizer, T5ForConditionalGeneration
from peft import PeftModel
from pathlib import Path

BASE = "Salesforce/codet5p-220m-py"
CKPT = "codebridge/models/codet5p_lora/checkpoint-150"
OUT  = "codebridge/models/codet5p_220m_merged_ckpt150"

tok = AutoTokenizer.from_pretrained(BASE)
base = T5ForConditionalGeneration.from_pretrained(BASE)
model = PeftModel.from_pretrained(base, CKPT)
merged = model.merge_and_unload()

Path(OUT).mkdir(parents=True, exist_ok=True)
merged.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("Saved merged model to:", OUT)