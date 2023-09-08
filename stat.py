import pickle
import matplotlib.pyplot as plt

pickle_path = 'D:/save.pkl'
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

def dep_vs_attn_mean(dep_dists, attn_dists, lay, head):
	attn_dist = [i[lay,head].item() for i in attn_dists]

	fig = plt.figure()
	ax = fig.subplots()

	ax.scatter(dep_dists, attn_dist, facecolors='none', edgecolors='k',linewidths=0.5)
	ax.set_xlabel('sentence standard mean dependency distance')
	ax.set_ylabel(f'sentence standard mean attention distance at head {lay+1:02d}-{head+1:02d}')
	plt.show()