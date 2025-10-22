from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,DataCollatorWithPadding
import evaluate
import numpy as np

pre_model="train_model/checkpoint-423"
tokenizer="train_model/checkpoint-423"
data_files="balanced_dataset.csv"
output_dir="./train_model"
# Step 1: Load the dataset
dataset = load_dataset("csv", data_files=data_files)
# 字段检测与过滤
required_fields = ["review", "label"]
def is_valid(example):
    for field in required_fields:
        if field not in example or example[field] is None:
            return False
        if field == "review" and not isinstance(example[field], str):
            return False
        if field == "label":
            try:
                label_int = int(example[field])
                if label_int not in [0, 1, 2]:
                    return False
            except Exception:
                return False
    return True

dataset = dataset.filter(is_valid)
# Split the dataset into training and validation sets
split_dataset = dataset["train"].train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
valid_dataset = split_dataset["test"]

print("Training dataset size:", len(train_dataset))
print("Validation dataset size:", len(valid_dataset))

# Step 2: Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer)

def preprocess_function(examples):
    res = tokenizer(
        examples["review"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    # 确保 labels 是一维 int 列表
    if isinstance(examples.get("label"), list):
        res["labels"] = [int(x) for x in examples["label"]]
    else:
        res["labels"] = [int(examples["label"])]
    return res

# 预处理训练与验证集
tokenized_train = train_dataset.map(
    preprocess_function, batched=True, remove_columns=["label", "review"]
)
tokenized_valid = valid_dataset.map(
    preprocess_function, batched=True, remove_columns=["label", "review"]
)

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(pre_model, num_labels=3)

# 评估指标
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy.compute(predictions=preds, references=labels)
    f1_macro = f1.compute(predictions=preds, references=labels, average="macro")
    prec = precision.compute(predictions=preds, references=labels, average="macro")
    rec = recall.compute(predictions=preds, references=labels, average="macro")
    return {
        "accuracy": acc["accuracy"],
        "macro_f1": f1_macro["f1"],
        "precision": prec["precision"],
        "recall": rec["recall"],
    }


# Step 4: Set up training arguments（回退使用旧版 evaluate_during_training，其他尽量保持不变）
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    fp16=True,
    # 日志与复现
    logging_dir="./logs",
    logging_steps=25,

)

# Step 5: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    
)

# Step 6: Train the model
trainer.train()

# Step 7: Evaluate the model
results = trainer.evaluate(tokenized_valid)
print("Evaluation results:", results)

## 已按你的需求移除基线对比，保留纯训练与评估