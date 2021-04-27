import networkx as nx
from fa2 import ForceAtlas2
import matplotlib.pyplot as plt

# 30.167982550917607 nch-nsu:

node2color = {
	'nhe': 'green', 'azz': 'orange', 'nch': 'green', 'ncj': 'orange',
	'ncl': 'green', 'ngu': 'orange', 'nhi': 'orange',
	'nhw': 'green', 'nhx': 'blue', 'nhy': 'orange', 'nsu': 'orange'
}

G=nx.Graph()

nn = set()
ee = []
labels = {}
for line in open('../variants_no_nhx.tsv').readlines():
	line = line.strip()
	if line == '':
		continue
	row = line.split(' ')
	l1 = row[1].split('-')[0]
	l2 = row[1].split('-')[1].strip(':')
	pp = 1.0 - float(row[0])

	nn.add(l1)
	nn.add(l2)
	labels[l1] = l1
	labels[l2] = l2
	ee.append((l1, l2, pp))

#for l1 in nn:
#	G.add_node(l1, label=l1)

for (l1, l2, pp) in ee:
	G.add_edge(l1,l2,weight=pp)

forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=True,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=False,
                        gravity=1.0,

                        # Log
                        verbose=True)


positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=25000)
# nx.draw_networkx(G, positions, with_labels=True, node_size=400, node_color="blue", alpha=0.4, font_weight='bold', font_size=10) # node_size=20
nx.draw_networkx_nodes(G, positions, node_size=400, node_color=[node2color[n] for n in G.nodes], alpha=0.4)  # node_color="blue"
nx.draw_networkx_edges(G, positions, edge_color="blue", alpha=0.05)
nx.draw_networkx_labels(G, positions, labels, font_weight='bold', font_size=12, font_color='k', alpha=1.0)
plt.axis('off')
plt.savefig('nahuatl-force-atlas-color-trial_no_nhx.pdf')
plt.show()
