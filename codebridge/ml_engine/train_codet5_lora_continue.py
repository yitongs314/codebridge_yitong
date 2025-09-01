from pathlib import Path
from datasets import load_from_disk
from transformers import (AutoTokenizer, T5ForConditionalGeneration,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)
from peft import PeftModel
import loralib as lora

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "codebridge" / "data" / "processed" / "codet5p_tokenized"

BASE = "Salesforce/codet5p-220m-py"
CKPT = ROOT / "codebridge" / "models" / "codet5p_lora" / "checkpoint-150"

tok = AutoTokenizer.from_pretrained(BASE)
base = T5ForConditionalGeneration.from_pretrained(BASE)
model = PeftModel.from_pretrained(base, str(CKPT), is_trainable=True)   # 载入 LoRA 权重作为起点
lora.mark_only_lora_as_trainable(model)                   # <-- 只放开 LoRA 参数
model.train()                                        # 进入训练模式

# 省内存：训练关 cache、开梯度检查点
model.config.use_cache = False
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

ds = load_from_disk(str(DATA_DIR))

from datasets import concatenate_datasets
train_ds = concatenate_datasets([ds["train"], ds["validation"]]).shuffle(seed=42)
eval_ds  = ds["validation"].select(range(min(200, len(ds["validation"])))) 

collator = DataCollatorForSeq2Seq(tok, model=model)

args = Seq2SeqTrainingArguments(
    output_dir=str(ROOT/"codebridge"/"models"/"codet5p_lora_ft_val_v2"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    max_grad_norm=0.5,
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    optim="adamw_torch",
    predict_with_generate=False,
    dataloader_pin_memory=False,
    report_to="none",
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable} / {total}")


trainer = Seq2SeqTrainer(model=model, args=args,
                         train_dataset=train_ds, eval_dataset=eval_ds,
                         data_collator=collator, tokenizer=tok)
trainer.train()