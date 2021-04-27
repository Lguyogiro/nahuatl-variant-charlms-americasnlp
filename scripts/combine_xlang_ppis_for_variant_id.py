import glob
import pandas as pd
from sklearn.metrics import classification_report


files = glob.glob("../variantes-del-nahuatl/scripts/EVAL_PPs-*.txt")
langs = [f.split('/')[-1].split('-')[-1][:-4] for f in files]
columns = {}
for i in range(len(files)):
    lang = langs[i]
    with open(files[i]) as f:
        data = [float(line.strip('\n')) for line in f]
        columns[lang] = data

df = pd.DataFrame(columns)
truth = pd.read_csv('../variantes-del-nahuatl/data/variant_id/char/normalized/eval.tsv', sep='\t', header=None)
df['label'] = truth[1]
df.to_csv('../ppi_features_variant_id_test_set.csv', index=False)
predictions = df[[c for c in df.columns if c != 'label']].idxmax(axis=1)
labels = df.label.tolist()
predictions = predictions.tolist()

print(classification_report(labels, predictions))