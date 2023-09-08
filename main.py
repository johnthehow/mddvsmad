from thehow.transeasy.bert import bertplus_hier
from thehow.tuda import depd_core
import pickle

conll_path = 'D:/test/en_pud-ud-test.conllu'


trees = depd_core.trees_gi(conll_path)


def standard(trees):
	save_path = 'D:/test/standard.pkl'
	tree_cnt = 0

	dep_distances = []
	attn_distances = []

	while True:
		try:
			print(f'processing {tree_cnt}-th tree',end='\x1b\r')
			tree = next(trees)
			dep_dist = tree.depd_std_mean_abs
			dep_distances.append(dep_dist)
			attn_dist = bertplus_hier.analyzer(tree.text_lower, tree.tokens_lower).attentions.noclssep.scale.linear.reduced.standard_attention_distance
			attn_distances.append(attn_dist)
			tree_cnt += 1
		except StopIteration:
			print('done')
			break

	res = [dep_distances, attn_distances]

	with open(save_path, mode='wb') as file:
		pickle.dump(res, file)

def raw(trees):
	save_path = 'D:/test/raw.pkl'
	tree_cnt = 0
	dep_distances = []
	attn_distances = []

	while True:
		try:
			print(f'processing {tree_cnt}-th tree',end='\x1b\r')
			tree = next(trees)
			dep_dist = tree.depd_std_mean_abs
			dep_distances.append(dep_dist)
			attn_dist = bertplus_hier.analyzer(tree.text_lower, tree.tokens_lower).attentions.noclssep.scale.linear.reduced.attention_distance
			attn_distances.append(attn_dist)
			tree_cnt += 1
		except StopIteration:
			print('done')
			break

	res = [dep_distances, attn_distances]

	with open(save_path, mode='wb') as file:
		pickle.dump(res, file)


ss = raw(trees)