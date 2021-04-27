import sys, numpy as np
#from sklearn.utils.extmath import softmax
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

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

font = {'family' : 'FreeSans',
        'size'   : 48}

plt.rc('font', **font)


plt.figure(figsize=(25, 12))
plt.title('Variant Clustering by Cross-Variant Perplexity')
plt.ylabel('Language')
plt.xlabel('Distance')
dendrogram(
    Z,
    orientation='left',
    labels=langs,
#    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=24,  # font size for the x axis labels
)

plt.tight_layout()
#plt.show()
plt.savefig('dendrogram_FULL_RAWPPI_25x12-no-nhx.pdf', orientation='landscape')