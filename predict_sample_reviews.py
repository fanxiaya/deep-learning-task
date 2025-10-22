from transformers import pipeline
import pandas as pd
import re

# ========= 可配置区域 =========
CSV_PATH = 'test.csv'  # 待评估数据（需包含 review, label 两列）
MODEL_PATH = 'bert-base-chinese'  # 待评估的单个模型
BATCH_SIZE = 64  # 推理批大小
# ============================


def to_int_label(x):
	try:
		v = int(x)
		return v if v in (0, 1, 2) else None
	except Exception:
		return None


def parse_label_from_str(label_str: str) -> int | None:
	"""将 pipeline 返回的 label 字符串解析为数字（优先匹配末尾数字）。"""
	if label_str is None:
		return None
	m = re.search(r'(?:LABEL[_\-])?(\d+)$', str(label_str))
	if m:
		try:
			return int(m.group(1))
		except Exception:
			return None
	s = str(label_str).lower()
	if 'negative' in s:
		return 0
	if 'neutral' in s:
		return 1
	if 'positive' in s:
		return 2
	return None


def main():
	# 1) 读取与清洗数据
	df = pd.read_csv(CSV_PATH)
	df = df[df['review'].apply(lambda x: isinstance(x, str))].copy()
	df['label_int'] = df['label'].apply(to_int_label)
	df = df[df['label_int'].notna()].copy()
	df['label_int'] = df['label_int'].astype(int)

	reviews = df['review'].tolist()
	labels = df['label_int'].tolist()

	print(f"评估样本数: {len(labels)}  | 来自: {CSV_PATH}")

	# 2) 使用单个模型预测
	clf = pipeline('text-classification', model=MODEL_PATH, tokenizer=MODEL_PATH)
	preds = clf(reviews, truncation=True, batch_size=BATCH_SIZE)

	# 仅使用 argmax 的 label 作为预测类别
	pred_labels = []
	for pred in preds:
		if isinstance(pred, dict):
			lbl = parse_label_from_str(pred.get('label'))
			pred_labels.append(int(lbl if lbl is not None else 0))
		else:
			pred_labels.append(0)

	# 3) 计算总体准确率
	total = len(labels)
	correct = sum(int(p == t) for p, t in zip(pred_labels, labels))
	acc = correct / total if total else 0.0

	print(f"使用的模型: {MODEL_PATH}")
	print(f"评估样本数: {total}")
	print(f"总的正确率: {acc:.4f}  (正确 {correct} / 总计 {total})")


if __name__ == '__main__':
	main()

