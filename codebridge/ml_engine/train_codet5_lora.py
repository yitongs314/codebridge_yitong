from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, T5ForConditionalGeneration,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)
from peft import LoraConfig, TaskType, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "codebridge" / "data" / "processed" / "codet5p_tokenized"
CKPT = "Salesforce/codet5p-220m-py" #-py python only

tok = AutoTokenizer.from_pretrained(CKPT)
ds  = load_from_disk(str(DATA_DIR))

model = T5ForConditionalGeneration.from_pretrained(CKPT)
model.config.use_cache = False            
model.gradient_checkpointing_enable()    

# lora setup
lora = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=["q", "v"], 
    bias="none",
)
model = get_peft_model(model, lora)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

collator = DataCollatorForSeq2Seq(tok, model=model)

args = Seq2SeqTrainingArguments(
    output_dir=str(ROOT / "codebridge" / "models" / "codet5p_lora"),
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=64,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=False,
    dataloader_pin_memory=False,
    optim="adafactor", 
    bf16=False, fp16=False,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model, args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=collator, tokenizer=tok,
)
print("device =", model.device, "| mps_available =", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
trainer.train()