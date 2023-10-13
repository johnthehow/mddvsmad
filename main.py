from thehow.transeasy.bert import bertplus_hier
from thehow.tuda import depd_core
import pickle

conll_path = 'D:/test/en_pud-ud-test.conllu'


trees = depd_core.trees_gi(conll_path)


def get_sent_mdd_and_mad_mean_abs_std(trees): # 获取trees树库中所有句子的句平均依存距离(绝对值+句长标准化)和注意力距离(绝对值+句长标准化)
	save_path = 'D:/test/standard.pkl'
	tree_cnt = 0

	dep_distances = []
	attn_distances = []

	while True:
		try:
			print(f'processing {tree_cnt}-th tree',end='\x1b\r')
			tree = next(trees)
			dep_dist = tree.depd_mean_abs_std
			dep_distances.append(dep_dist)
			attn_dist = bertplus_hier.analyzer(tree.text_lower, tree.tokens_lower).attentions.noclssep.scale.linear.reduced.standard_attention_distance_abs
			attn_distances.append(attn_dist)
			tree_cnt += 1
		except StopIteration:
			print('done')
			break

	res = [dep_distances, attn_distances]

	with open(save_path, mode='wb') as file:
		pickle.dump(res, file)

def get_sent_mdd_mean_directed_and_mad_mean_abs(trees):
	save_path = 'D:/test/raw.pkl'
	tree_cnt = 0
	dep_distances = []
	attn_distances = []

	while True:
		try:
			print(f'processing {tree_cnt}-th tree',end='\x1b\r')
			tree = next(trees)
			dep_dist = tree.depd_mean_directed
			dep_distances.append(dep_dist)
			attn_dist = bertplus_hier.analyzer(tree.text_lower, tree.tokens_lower).attentions.noclssep.scale.linear.reduced.attention_distance_abs
			attn_distances.append(attn_dist)
			tree_cnt += 1
		except StopIteration:
			print('done')
			break

	res = [dep_distances, attn_distances]

	with open(save_path, mode='wb') as file:
		pickle.dump(res, file)

def get_sent_mdd_mean_raw_and_mad_mean_directed(trees):
	save_path = 'D:/test/raw_directed.pkl'
	tree_cnt = 0
	dep_distances = []
	attn_distances = []

	while True:
		try:
			print(f'processing {tree_cnt}-th tree',end='\x1b\r')
			tree = next(trees)
			dep_dist = tree.depd_mean_directed
			dep_distances.append(dep_dist)
			attn_dist = bertplus_hier.analyzer(tree.text_lower, tree.tokens_lower).attentions.noclssep.scale.linear.reduced.attention_distance_directed
			attn_distances.append(attn_dist)
			tree_cnt += 1
		except StopIteration:
			print('done')
			break

	res = [dep_distances, attn_distances]

	with open(save_path, mode='wb') as file:
		pickle.dump(res, file)

result = get_sent_mdd_and_mad_mean_directed(trees)