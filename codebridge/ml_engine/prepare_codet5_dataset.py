# pip install -U datasets transformers accelerate


from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from pathlib import Path

CKPT = "Salesforce/codet5p-220m-py"
RAW_CSV = "data/python_code_data.csv"
OUT_DIR = "data/processed/codet5p_tokenized"  # 保存 tokenized 数据集（arrow 格式）

# 2) 载入 tokenizer
tok = AutoTokenizer.from_pretrained(CKPT)

# 3) 载入原始 CSV（两列：text, code）
#    注意：我们会先做简单清洗（去空值），再切分 train/val/test

raw_all = load_dataset("csv", data_files={"train": RAW_CSV})["train"]
raw_all = raw_all.filter(lambda ex: (ex.get("text") is not None) and (ex.get("code") is not None))

# 4) 切分（示例：80/10/10）
tmp = raw_all.train_test_split(test_size=0.2, seed=42)
val_test = tmp["test"].train_test_split(test_size=0.5, seed=42)
raw = DatasetDict(train=tmp["train"], validation=val_test["train"], test=val_test["test"])

# 5) 构建训练用的「文字提示 → 代码」格式
def build_prompt(ex):
    # 和你推理时一致：明确指定语言、说明任务、约定输出位置
    return f"Generate Python code:\nTask: {ex['text']}\nCode:"

# 6) tokenization：把输入与标签分别转成 token id
def to_features(ex):
    X = tok(build_prompt(ex), max_length=384, truncation=True)
    # T5 的标签放在 "labels"；DataCollator 会把 padding 的位置改成 -100 以忽略 loss
    with tok.as_target_tokenizer():
        y = tok(ex["code"], max_length=256, truncation=True)
    X["labels"] = y["input_ids"]
    return X

tok_ds = raw.map(to_features, remove_columns=raw["train"].column_names)

# 7) 保存到磁盘，训练时可用 load_from_disk 直接载入
Path(OUT_DIR).parent.mkdir(parents=True, exist_ok=True)
tok_ds.save_to_disk(OUT_DIR)

# 8) 小小 sanity check
print(tok_ds)
print("Example prompt:\n", build_prompt(raw["train"][0])[:400], "...\n")