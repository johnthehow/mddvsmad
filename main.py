from thehow.transeasy.bert import bertplus_hier
from thehow.tuda import depd_core
import pickle

conll_path = 'D:/test/en_pud-ud-test.conllu'
save_path = 'D:/test/save.pkl'

trees = depd_core.trees_gi(conll_path)

tree_cnt = 0

dep_distances = []
attn_distances = []

while True:
	try:
		tree = next(trees)
		dep_dist = tree.depd_std_mean_abs
		dep_dists.append(dep_dist)
		attn_dist = bertplus_hier.analyzer(tree.text_lower, tree.tokens_lower).attentions.noclssep.scale.linear.reduced.standard_attention_distance
		attn_dists.append(attn_dist)
	except StopIteration:
		break

res = [depd_distances, attn_distances]

with open(save_path, mode='wb') as file:
	pickle.dump(res, file)