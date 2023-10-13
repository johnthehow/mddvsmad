from thehow.transeasy.bert import bertplus_hier
from thehow.tuda import depd_core
from thehow.snips.logx import logger
from thehow.snips import timex
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('ismean')
parser.add_argument('isabs')
parser.add_argument('isstd')
args = parser.parse_args()



def sent_mdd_vs_mad(trees, variation, pkl_savepath, fig_savepath):
	pkl_savepath_obj = Path(pkl_savepath)
	fig_savepath_obj = Path(fig_savepath)
	tree_cnt = 0
	dep_distances = []
	attn_distances = []

	while True:
		try:
			# print(f'processing {tree_cnt}-th tree',end='\x1b\r')
			logger.info(f'processing {tree_cnt}-th tree')
			tree = next(trees)
			if variation == ['mean','directed','standard']:
				sent_dep_distance = tree.depd_mean_directed_std
				sent_attn_distance = bertplus_hier.analyzer(tree.text_lower, tree.tokens_lower).attentions.noclssep.scale.linear.reduced.attention_distance_mean_directed_std
			elif variation == ['mean','directed','raw']:
				sent_dep_distance = tree.depd_mean_directed
				sent_attn_distance = bertplus_hier.analyzer(tree.text_lower, tree.tokens_lower).attentions.noclssep.scale.linear.reduced.attention_distance_mean_directed
			elif variation == ['mean','abs','standard']:
				sent_dep_distance = tree.depd_mean_abs_std
				sent_attn_distance = bertplus_hier.analyzer(tree.text_lower, tree.tokens_lower).attentions.noclssep.scale.linear.reduced.attention_distance_mean_abs_std
			elif variation == ['mean','abs','raw']:
				sent_dep_distance = tree.depd_mean_abs
				sent_attn_distance = bertplus_hier.analyzer(tree.text_lower, tree.tokens_lower).attentions.noclssep.scale.linear.reduced.attention_distance_mean_abs
			else:
				raise ValueError('[ERR] unsupported varation combination')
			dep_distances.append(sent_dep_distance)
			attn_distances.append(sent_attn_distance)
			tree_cnt += 1
		except StopIteration:
			logger.info('Done')
			break
	res = [dep_distances, attn_distances]

	with open(pkl_savepath, mode='wb') as file:
		pickle.dump(res, file)
	logger.info(f'result pickle saved at {pkl_savepath}')

	with open(pkl_savepath, mode='rb') as file:
		res_input = pickle.load(file)
	logger.info(f'start of visualization')

	dep_dists = res_input[0] # List(1000,)
	attn_dists = res_input[1] # List(1000, Tensor(12,12))

	fig_cnt = 0
	for lay in range(12):
		for head in range(12):
			attn_dist = [i[lay,head].item() for i in attn_dists] # List[1000]
			fig = plt.figure()
			ax = fig.subplots()			
			ax.scatter(dep_dists, attn_dist, facecolors='none', edgecolors='k',linewidths=0.5)
			ax.set_xlabel(f'sentence dependency distance {variation[0]} {variation[1]} {variation[2]}')
			ax.set_ylabel(f'sentence attention distance {variation[0]} {variation[1]} {variation[2]} {lay+1:02d}-{head+1:02d}')
			ax.set_title(f'MDD(Sentence) vs MAD(Sentence) at Head {lay+1:02d}-{head+1:02d}')
			filename = f'{lay+1:02d}_{head+1:02d}.png'
			plt.savefig(fig_savepath_obj.joinpath(filename),format='png')
			plt.close()
			fig_cnt += 1
			logger.info(f'drawn {fig_cnt} figs')
	logger.info(f'figs saved at {fig_savepath}')


if __name__ == '__main__':
	variation = [args.ismean,args.isabs,args.isstd]
	# variation = ['mean','directed','raw']
	conll_path = 'D:/test/mddvsmad/corpus/en_pud-ud-test.conllu'
	pkl_savepath = f'D:/test/mddvsmad/pkl/result_{timex.timestamp14()}.pkl'
	fig_savepath = f'D:/test/mddvsmad/fig/{variation[0]}_{variation[1]}_{variation[2]}_{timex.timestamp14()}'
	os.makedirs(fig_savepath)
	trees = depd_core.trees_gi(conll_path)
	logger.info(f'setting: {variation}')
	result = sent_mdd_vs_mad(trees, variation , pkl_savepath, fig_savepath)
	logger.info(f'log saved at: {logger.handlers[1].baseFilename}')