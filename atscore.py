from thehow.transeasy.bert import bertplus_hier
from thehow.tuda import depd_core
from thehow.snips.logx import logger
from thehow.snips import timex
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os
import argparse


def word_std_abs_dd_vs_attn_score(trees, pkl_savepath):
	pkl_savepath_obj = Path(pkl_savepath)
	word_std_abs_dds = []
	word_attn_scores_word_to_head = []
	word_attn_scores_head_to_word = []
	tree_cnt = 1
	while True:
		try:
			logger.info(f'processing {tree_cnt}-th tree')
			tree = next(trees)
			tree_len = tree.len-1 # root词不计依存距离, n个词就有n-1个依存距离
			try:
				tree_attn_matrices = bertplus_hier.analyzer(tree.text_lower, tree.tokens_lower).attentions.noclssep.scale.linear.reduced.matrices # tensor(12,12,句长,句长)
				assert tree_attn_matrices.shape[2] == tree.len, logger.error('Attention matrix size not equal to ud tree length')
				for node in tree.nodes:
					if node.headid != 0: # 跳过root词的依存距离和注意力值提取
						node_std_abs_dd = abs(node.headid-node.id)/tree_len
						word_std_abs_dds.append(node_std_abs_dd)
						word_attn_score_word_to_head = tree_attn_matrices[:,:,node.id-1,node.headid-1].detach().numpy() # numpy.ndarray(12,12)
						word_attn_scores_word_to_head.append(word_attn_score_word_to_head)
						word_attn_score_head_to_word = tree_attn_matrices[:,:,node.headid-1,node.id-1].detach().numpy() # numpy.ndarray(12,12)
						word_attn_scores_head_to_word.append(word_attn_score_head_to_word)
			except ValueError:
				logger.error(f'Value error occured at {tree_cnt}-th sentence "{tree.text}", passed.')
			except IndexError:
				logger.error(f'Index error occured at {tree_cnt}-th sentence "{tree.text}", passed')
			except AssertionError:
				logger.error(f'Assertion error occured at {tree_cnt}-th sentence "{tree.text}", passed')
			tree_cnt += 1
		except StopIteration:
			logger.info('Done')
			break
	res = [word_std_abs_dds, word_attn_scores_word_to_head, word_attn_scores_head_to_word]

	final_savepath_obj = pkl_savepath_obj.joinpath(f'result_{timex.timestamp14()}.pkl')
	with open(final_savepath_obj, mode='wb') as file:
		pickle.dump(res, file)
		logger.info(f'pickle saved at {final_savepath_obj}')
	return final_savepath_obj # 结果pkl路径, 用于后续的可视化

def viz(pkl_path, fig_savepath):
	fig_savepath_obj = Path(fig_savepath)
	timed_fig_w2h_savepath_obj = fig_savepath_obj.joinpath(f'word_to_head_{timex.timestamp14()}')
	timed_fig_h2w_savepath_obj = fig_savepath_obj.joinpath(f'head_to_word_{timex.timestamp14()}')
	timed_fig_2way_savepath_obj = fig_savepath_obj.joinpath(f'two_way_{timex.timestamp14()}')
	os.makedirs(timed_fig_w2h_savepath_obj)
	os.makedirs(timed_fig_h2w_savepath_obj)
	os.makedirs(timed_fig_2way_savepath_obj)
	with open(pkl_path, mode='rb') as file:
		res_input = pickle.load(file)
	logger.info(f'start of visualization')

	word_std_abs_dds = res_input[0] # List(*)
	word_attn_scores_word_to_head = res_input[1] # List(*, numpy.ndarray(12,12))
	word_attn_scores_head_to_word = res_input[2] # List(*, numpy.ndarray(12,12))


	# 绘图: 词依存距离(绝对值_句长标准化)vs 从属词到支配词的注意力值
	fig_cnt = 0
	for lay in range(12):
		for head in range(12):
			word_attn_score_word_to_head = [i[lay,head] for i in word_attn_scores_word_to_head] # List[*]
			fig = plt.figure()
			ax = fig.subplots()
			ax.scatter(word_std_abs_dds, word_attn_score_word_to_head, facecolors='none', edgecolors='k',linewidths=0.5)
			ax.set_xlabel(f'word dependency distances abs std')
			ax.set_ylabel(f'word attention scores word to head')
			ax.set_title(f'word depd(abs+std) vs word atnscore(word2head) at Head {lay+1:02d}-{head+1:02d}')
			filename = f'{lay+1:02d}_{head+1:02d}.png'
			plt.savefig(timed_fig_w2h_savepath_obj.joinpath(filename),format='png')
			plt.cla()
			plt.clf()
			plt.close()
			fig_cnt += 1
			logger.info(f'drawn {fig_cnt} figs')
	logger.info(f'figs saved at {timed_fig_w2h_savepath_obj}')

	# 绘图: 词依存距离(绝对值_句长标准化)vs 支配词到从属词的注意力值
	fig_cnt = 0
	for lay in range(12):
		for head in range(12):
			word_attn_score_head_to_word = [i[lay,head] for i in word_attn_scores_head_to_word] # List[*]
			fig = plt.figure()
			ax = fig.subplots()
			ax.scatter(word_std_abs_dds, word_attn_score_head_to_word, facecolors='none', edgecolors='k',linewidths=0.5)
			ax.set_xlabel(f'word dependency distances abs std')
			ax.set_ylabel(f'word attention scores head to word')
			ax.set_title(f'word depd(abs+std) vs word atnscore(head2word) at Head {lay+1:02d}-{head+1:02d}')
			filename = f'{lay+1:02d}_{head+1:02d}.png'
			plt.savefig(timed_fig_h2w_savepath_obj.joinpath(filename),format='png')
			plt.cla()
			plt.clf()
			plt.close()
			fig_cnt += 1
			logger.info(f'drawn {fig_cnt} figs')
	logger.info(f'figs saved at {timed_fig_h2w_savepath_obj}')

	# 绘图: 词依存距离(绝对值_句长标准化) vs (支配词到从属词的注意力值+从属词到支配词的注意力值)/2
	fig_cnt = 0
	for lay in range(12):
		for head in range(12):
			word_attn_score_head_to_word = [i[lay,head] for i in word_attn_scores_head_to_word] # List[*]; i:(12,12)
			word_attn_score_word_to_head = [i[lay,head] for i in word_attn_scores_word_to_head] # List[*]; i:(12,12)
			word_attn_score_2way = [(i+j)/2 for i,j in zip(word_attn_score_head_to_word, word_attn_score_word_to_head)] # List[*]
			fig = plt.figure()
			ax = fig.subplots()
			ax.scatter(word_std_abs_dds, word_attn_score_2way, facecolors='none', edgecolors='k',linewidths=0.5)
			ax.set_xlabel(f'word dependency distances abs std')
			ax.set_ylabel(f'word attention scores two way')
			ax.set_title(f'word depd(abs+std) vs word atnscore(two way) at Head {lay+1:02d}-{head+1:02d}')
			filename = f'{lay+1:02d}_{head+1:02d}.png'
			plt.savefig(timed_fig_2way_savepath_obj.joinpath(filename),format='png')
			plt.cla()
			plt.clf()
			plt.close()
			fig_cnt += 1
			logger.info(f'drawn {fig_cnt} figs')
	logger.info(f'figs saved at {timed_fig_2way_savepath_obj}')


if __name__ == '__main__':
	conll_path = 'D:/test/atscore/corpus/en_pud-ud-test.conllu'
	pkl_savepath = 'D:/test/atscore/pkl'
	trees = depd_core.trees_gi(conll_path)
	result_pkl = word_std_abs_dd_vs_attn_score(trees, pkl_savepath)
	logger.info(f'log saved at: {logger.handlers[1].baseFilename}')
	fig_savepath = 'D:/test/atscore/fig/'
	viz(result_pkl, fig_savepath)