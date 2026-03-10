import json, re
from collections import Counter

results = []
with open('results/Llama_3_2_3B/OpenTSLMSP_moment/stage4_sleep_cot/results/test_predictions.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        results.append(obj)

total = len(results)
correct = 0
gold_labels = []
pred_labels = []

def extract_label(text):
    text = text.replace('<|end_of_text|>', '').strip()
    match = re.findall(r'[Aa]nswer:\s*(.+)', text)
    if match:
        label = match[-1].strip().rstrip('.').lower()
        return label
    return text[-50:].strip().lower()

for r in results:
    gold = extract_label(r['gold'])
    pred = extract_label(r['generated'])
    gold_labels.append(gold)
    pred_labels.append(pred)
    if gold == pred:
        correct += 1

accuracy = correct / total
print(f'Total samples: {total}')
print(f'Correct: {correct}')
print(f'Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)')
print()

print('Gold label distribution:')
for label, count in sorted(Counter(gold_labels).items(), key=lambda x: -x[1]):
    print(f'  {label}: {count}')

print()
print('Pred label distribution:')
for label, count in sorted(Counter(pred_labels).items(), key=lambda x: -x[1]):
    print(f'  {label}: {count}')

# Per-class metrics
print()
print('Per-class metrics:')
all_classes = sorted(set(gold_labels))
f1_scores = []
for cls in all_classes:
    tp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == cls and g == cls)
    fp = sum(1 for p, g in zip(pred_labels, gold_labels) if p == cls and g != cls)
    fn = sum(1 for p, g in zip(pred_labels, gold_labels) if p != cls and g == cls)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    f1_scores.append(f1)
    print(f'  {cls:20s}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})')

macro_f1 = sum(f1_scores) / len(f1_scores)
print(f'\nMacro F1: {macro_f1:.4f}')
