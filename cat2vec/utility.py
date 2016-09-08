from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_with_labels(final_embeddings, labels, reverse_dictionary,
                     filename='tsne.png'):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.show()


def load_data(debug=False, dataset='ipinyou'):
    base_path = './data/' + dataset
    if debug:
        path = './data/training_debug.csv'
        # base_path = './data'
    else:
        path = base_path + '/training_not_aligned.csv'
    f = open(path, 'r')
    raw_data = []
    for line in f.readlines():
        raw_data.append([int(x) for x in line.strip().split(',')])
    vocabulary_size = 0
    reverse_dictionary_raw = np.array(pd.read_csv(
        base_path + '/reverse_dictionary_not_aligned.csv', sep=',', header=None))
    reverse_dictionary = {}
    dictionary = {}
    for item in reverse_dictionary_raw:
        reverse_dictionary[int(item[1])] = item[0]
        dictionary[item[0]] = int(item[1])
    if item[1] > vocabulary_size:
        vocabulary_size = item[1]
    vocabulary_size = len(dictionary.keys())
    return raw_data, reverse_dictionary, dictionary, vocabulary_size
