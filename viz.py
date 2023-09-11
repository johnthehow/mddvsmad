import pickle
import matplotlib.pyplot as plt
from pathlib import Path

pickle_path = 'D:/raw.pkl'
with open(pickle_path, mode='rb') as file:
	res = pickle.load(file)

dep_dists = res[0]
attn_dists = res[1]


def dep_vs_attn_mean_144(dep_dists, attn_dists):
	attn_dists_means = [i.mean().item() for i in attn_dists]

	fig = plt.figure()
	ax = fig.subplots()

	ax.scatter(dep_dists, attn_dists_means, facecolors='none', edgecolors='k',linewidths=0.5)
	ax.set_xlabel('sentence standard mean dependency distance')
	ax.set_ylabel('sentence standard mean attention distance')
	plt.show()
	plt.close()

def dep_vs_attn_mean(dep_dists, attn_dists, save_path):
	save_path_obj = Path(save_path)
	for lay in range(12):
		for head in range(12):
			attn_dist = [i[lay,head].item() for i in attn_dists]
			fig = plt.figure()
			ax = fig.subplots()
			ax.scatter(dep_dists, attn_dist, facecolors='none', edgecolors='k',linewidths=0.5)
			ax.set_xlabel('sentence mean dependency distance')
			ax.set_ylabel(f'sentence mean attention distance {lay+1:02d}-{head+1:02d}')
			ax.set_title(f'MDD(Sentence) vs MAD(Sentence) at Head {lay+1:02d}-{head+1:02d}')
			filename = f'{lay+1:02d}_{head+1:02d}.png'
			plt.savefig(save_path_obj.joinpath(filename),format='png')
			plt.close()


dep_vs_attn_mean(dep_dists, attn_dists, 'D:/test/figs_raw/')

	