import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


dic = {}
langs = set()

for line in open('variants_no_nhx.tsv').readlines():
	line = line.strip()
	if line == '':
		continue
	row = line.split(' ')
	l1 = row[1].split('-')[0]
	l2 = row[1].split('-')[1].strip(':')
	pp = float(row[0])

	langs.add(l1)
	if l1 not in dic:
		dic[l1] = {}

	dic[l1][l2] = pp


langs = list(langs)
langs.sort()

X = []
for l1 in langs:
	X.append([dic[l1][i] for i in langs])


XX = np.array(X)

print(XX)

Z = linkage(XX, 'ward')

font = {'family': 'FreeSans',
		'size': 48}

plt.rc('font', **font)


plt.figure(figsize=(25, 12))
plt.title('Variant Clustering by Cross-Variant Perplexity')
plt.ylabel('Language')
plt.xlabel('Distance')
dendrogram(Z, orientation='left', labels=langs, leaf_font_size=24)

plt.tight_layout()
#plt.show()
plt.savefig('dendrogram_FULL_RAWPPI_25x12-no-nhx.pdf', orientation='landscape')