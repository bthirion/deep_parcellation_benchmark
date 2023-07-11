"""
Utilities for dictionary learning

Authors: Bertrand Thirion, Ana Luisa Pinho

Last update: June 2021

Compatibility: Python 3.5

"""

import os
import numpy as np
from sklearn.manifold import spectral_embedding
import matplotlib.pyplot as plt
from ibc_public.utils_data import CONTRASTS, all_contrasts
from joblib import Memory


def histogram_equalization(X):
    """Histogram equalization

    Parameters
    ----------
 
    X: list of arrays of shape (n_contrasts, n_voxels)
        the data to be equalized
 
    Returns
    -------
    X: list of arrays of shape (n_contrasts, n_voxels)
        the data after equalization
    """
    n_subjects = len(X)
    n_voxels = X[0].shape[1]
    n_contrasts = X[0].shape[0]
    for k in range(n_contrasts):
        h = np.zeros(n_voxels)
        for i in range(n_subjects):
            h += np.sort(X[i][k])
        h /= n_subjects
        for i in range(n_subjects):
            a = np.argsort(X[i][k])
            X[i][k][a] = h
    return X


def initial_dictionary(n_clusters, X,):
    """Creat initial dictionary"""
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0,
                             batch_size=200, n_init=10)
    kmeans = kmeans.fit(X.T)
    dictionary_ = kmeans.cluster_centers_
    dictionary = (dictionary_.T / np.sqrt((dictionary_ ** 2).sum(1))).T
    similarity = np.exp(np.corrcoef(dictionary))
    embedding = spectral_embedding(similarity, n_components=1)
    order = np.argsort(embedding.T).ravel()
    dictionary = dictionary[order]
    return dictionary


def make_dictionary(X, n_components=20, alpha=5., write_dir='/tmp/',
                    contrasts=[], method='multitask', l1_ratio=.5,
                    n_subjects=13):
    """Create dictionary + encoding"""
    from sklearn.decomposition import dict_learning_online, sparse_encode
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import MultiTaskLasso, MultiTaskElasticNet

    mem = Memory(write_dir, verbose=0)
    dictionary = mem.cache(initial_dictionary)(n_components, X)
    np.savez(os.path.join(write_dir, 'dictionary.npz'),
             loadings=dictionary, contrasts=contrasts)
    if method == 'online':
        components, dictionary = dict_learning_online(
                X.T, n_components, alpha=alpha,
                dict_init=dictionary,
                batch_size=200,
                method='cd',
                return_code=True,
                shuffle=True,
                n_jobs=1,
                positive_code=True)
        np.savez(os.path.join(write_dir, 'dictionary.npz'),
                 loadings=dictionary, contrasts=contrasts)
    elif method == 'sparse':
        components = sparse_encode(
            X.T, dictionary, alpha=alpha, max_iter=10, n_jobs=1,
            check_input=True, verbose=0, positive=True)
    elif method == 'multitask':
        # too many hard-typed parameters !!!
        n_voxels = X.shape[1] // n_subjects
        components = np.zeros((X.shape[1], n_components))
        clf = MultiTaskLasso(alpha=alpha)
        clf = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        for i in range(n_voxels):
            x = X[:, i: i + n_subjects * n_voxels: n_voxels]
            components[i: i + n_subjects * n_voxels: n_voxels] =\
                clf.fit(dictionary.T, x).coef_
    return dictionary, components


def cluster(Xr, n_components=20, write_dir='/tmp/', contrasts=[]):
    """Kmeans clustering"""
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_components, random_state=0,
                             batch_size=200, n_init=10)
    kmeans = kmeans.fit(Xr.T)
    dictionary = kmeans.cluster_centers_
    labels = kmeans.labels_
    n_samples = labels.size
    components = np.zeros((n_samples, n_components))
    components[np.arange(n_samples), labels] = 1
    similarity = np.exp(np.corrcoef(dictionary))
    embedding = spectral_embedding(similarity, n_components=1)
    order = np.argsort(embedding.T).ravel()
    dictionary = dictionary[order]
    components = components[:, order]
    np.savez(os.path.join(write_dir, 'dictionary.npz'),
             loadings=dictionary, contrasts=contrasts)
    return dictionary, components


def dictionary2labels(dictionary, task_list, path, facecolor=[.5, .5, .5],
                      contrasts=[], best_labels=[]):
    """Create a figure with labels reflecting the input dictionary"""
    from matplotlib.cm import gist_ncar
    LABELS = _make_labels(all_contrasts, task_list)
    w = dictionary
    n_contrasts = len(dictionary)
    plt.figure(facecolor=facecolor, figsize=(2.3, 6))
    # cmap = plt.get_cmap('gist_ncar')
    # colors = gist_ncar(np.linspace(0, 1, n_contrasts))
    colors = gist_ncar(
        np.linspace(0, 255, n_contrasts + 2).astype(np.int))[1:]
    
    for k, comp in enumerate(dictionary):
        if len(best_labels) >= len(dictionary):
            best_label = best_labels[k]
        else:
            weights = comp
            labels = [LABELS[contrast][x > 0] for (x, contrast) in
                      zip(comp, contrasts)]
            order = np.argsort(-weights)
            spec = weights == w.max(0)
            if spec.any():
                best_label = np.array(labels)[order][spec[order]][:2]
            else:
                best_label = labels[order[0]]
            print(best_label)
            best_label = np.unique(best_label)
            best_label = str(best_label).replace('[', '').replace(']', '')
            best_label = best_label.replace("' '", ", ")
            best_label = best_label.replace("'", "")
            best_labels.append(best_label)
        plt.text(0, .05 * k, best_label, weight='bold', color=colors[k],
                 fontsize=14)
    plt.axis('off')
    plt.subplots_adjust(left=.01, bottom=.01, top=.99, right=.99)
    plt.savefig(path, facecolor=facecolor, dpi=300)
    plt.show(block=False)
    return best_labels


def _make_labels(contrasts, task_list):
    labels = {}
    for i in range(len(CONTRASTS)):
        if CONTRASTS.task[i] in task_list:
            labels[CONTRASTS.contrast[i]] = [contrasts['negative label'][i],
                                             contrasts['positive label'][i]]
    return labels



def balanced_assignment(X, prototypes, weights=None, solver='sinkhorn',
                       reg=.1):
    """ Assigns X to weighted prototypes following optimal transport
    
    Parameters
    ----------
    X: array of shape (n_samples, n_features)
       the data to be aligned to prototypes
    prototypes: array of shape(n_prototypes, n_features)
       the targeted prototypes
    weights: array of shape (n_prototypes) or None
       the protoypes weights
    solver: str, optional
        the solver to use
    reg: float, optional
         entropic regularization

    """
    import ot
    from scipy.spatial.distance import cdist
    n = X.shape[0]
    p = prototypes.shape[0]
    a = np.ones(n) / n
    if weights is None:
        b = np.ones(p) / p
    else:
        b = weights
    
    M = cdist(X, prototypes)
    
    if solver == 'exact':
        R = ot.lp.emd(a, b, M) * n
    else:
        R = ot.sinkhorn(
            a, b, M, reg, method=solver) * n
    if (R == 0).all():
        stop
    return R, R.T.dot(X)


def balanced_clustering(X, groups, n_components=10, n_iterations=10, write_dir='/tmp',
                        tol=.01, reg=.1, dictionary=None, verbose=False):
    """ perform the clustering"""
    mem = Memory(write_dir, verbose=0)
    if dictionary is None:
        dictionary = mem.cache(initial_dictionary)(n_components, X.T)
    gvals = np.unique(groups)
    weights = np.ones(n_components) / n_components
    n_groups = len(gvals)

    inertia = np.infty
    for it in range(n_iterations):
        labels = []
        prototypes_ = np.zeros_like(dictionary)
        inertia_ = 0
        population = np.zeros(n_components)
        for group in gvals:
            R, Z = balanced_assignment(X[groups == group], dictionary, weights, reg=reg)
            labels_ = np.argmax(R, 1)
            #prototypes_ += np.array([np.mean(X[groups == group][labels_ == l], 0)
            #                         for l in np.unique(labels_)])
            for l in np.unique(labels_):
                population[l] += np.sum(labels_ == l)
                prototypes_[l] += np.sum(X[groups == group][labels_ == l], 0)
            inertia_ +=  np.sum((X[groups == group] - dictionary[labels_]) ** 2)
            labels.append(labels_)
        prototypes_ = (prototypes_.T /  population).T
        # prototypes_ /= n_groups
        dictionary = prototypes_
        labels = np.hstack(labels)
        if verbose:
            print(it, inertia_)
        if inertia_ > inertia - tol:
            break
        inertia = inertia_
    return dictionary, labels


def balanced_hierarchical_clustering(X, groups, n_components, n_iterations=10, tol=.01,
                                     reg=.1, write_dir='/tmp'):
    """ A hierrachical version of the clsutering""" 
    n_big_clusters = int(np.sqrt(n_components))
    _,  coarse_labels = balanced_clustering(
        X, groups, n_big_clusters, n_iterations, write_dir=write_dir,
        tol=tol, reg=reg)
    fine_labels = np.zeros_like(coarse_labels)
    q = 0
    global_dict = []
    for i in range(n_big_clusters):
        n_small_clusters = int(
            n_components * np.sum(coarse_labels == i) * 1. / X.shape[0])
        n_small_clusters = np.maximum(1, n_small_clusters)
        mask = coarse_labels == i
        local_dict,  local_labels = balanced_clustering(
            X[mask], groups[mask], n_small_clusters, n_iterations, write_dir=write_dir,
            tol=tol, reg=reg)
        fine_labels[mask] = q + local_labels
        q += n_small_clusters
        global_dict.append(local_dict)
    global_dict = np.vstack(global_dict)
    return global_dict, fine_labels


def test_balanced_clustering():
    """ test balanced clustering"""
    rng = np.random.RandomState(42)
    n_groups = 10
    n_samples = 50
    n_dim = 5
    delta = 1.
    X = []
    groups = []
    for group in range(n_groups):
        x = rng.randn(n_samples, n_dim) + rng.randn(n_dim)
        x[: int(n_samples / 2)] += delta
        X.append(x)
        groups.append([group] * n_samples)
        
    X = np.vstack(X)
    groups = np.hstack(groups)
    dictionary, labels = balanced_clustering(X, groups, n_components=2)
    print(dictionary)
    print(labels)
    import matplotlib.pyplot as plt
    plt.imshow(np.reshape(labels, (n_groups, n_samples)), interpolation='nearest')
    plt.axis('off')
    plt.show(block=False)



def plot_dictionary(fingerprint, contrasts, colors, write_name, close=True,
                      color_per_component=False, facecolor='k'):
    n_components = len(fingerprint)
    n_contrasts = len(contrasts)
    plt.figure(figsize=(6, 6), facecolor=facecolor)
    delta = .6 / n_components
    ax = plt.axes([0.01, 0, .39, 1], facecolor=facecolor)
    ax.axis('off')
    for j in range(n_contrasts):
        ax.text(1, .05 + .91 * j * 1./n_contrasts,
                contrasts[j].replace('_', ' '),
                color='w', fontsize=12, #fontweight='bold'
                ha='right'
        )
    for i in range(n_components):
        coef = fingerprint[i]
        ax = plt.axes([.4 + i * delta, 0, delta, 1])
        ax.axis('off')
        color = colors
        if color_per_component:
            color = colors[i]
        ax.barh(range(n_contrasts), coef, color=color, alpha=.99,
                ecolor=facecolor)
    if write_name is not None:
        plt.savefig(write_name)

        
def plot_dictionary_matrix(
        fingerprint, contrasts, colors, write_name, close=True,
        color_per_component=False, facecolor='k'):
    import seaborn as sns
    textcolor = 'w'
    if facecolor == 'w':
        textcolor = 'k'
    n_components = fingerprint.shape[1]
    n_contrasts = len(contrasts)
    plt.figure(figsize=(6, 6), facecolor=facecolor)
    ax = plt.axes([0.01, 0, .39, 1], facecolor=facecolor)
    for j in range(n_contrasts):
        ax.text(.99, .03 + .9 * j * 1./n_contrasts,
                contrasts[j].replace('_', ' '),
                color=textcolor, fontsize=12, #fontweight='bold'
                ha='right'
        )
    ax.axis('off')
    
    # plot the components colors
    ax = plt.axes([.4, 0.92, .59, 0.08])
    ax.imshow(np.arange(1, 1 + n_components)[np.newaxis],
               vmin=0, interpolation='nearest', cmap=plt.cm.gist_ncar)
    ax.axis('off')
    # plot the data
    ax = plt.axes([.4, 0.01, .59, .9])
    vmax = max( - fingerprint.min(), fingerprint.max())
    vmin = -vmax
    sns.heatmap(fingerprint, vmin=vmin, vmax=vmax, ax=ax, cbar=False,
                linecolor=facecolor, linewidth=3, cmap=plt.cm.RdBu_r,
                square=False,
    )
    ax.axis('off')
    if write_name is not None:
        plt.savefig(write_name)

        
def get_signature(X_, labels, clf, i):
    indicator = 2 * (labels == i) - (labels > 0)
    X = X_.T[indicator != 0]
    y = indicator[indicator != 0]
    clf.fit(X, y)
    return clf.coef_[0]


def multivariate_equalization(X_train, n_clusters, reg=.1, write_dir='/tmp'):
    """ learn and compensate shifts between datasets"""
    # perform a balanced clustering
    n_subjects, n_contrasts, n_nodes = X_train.shape
    features = np.reshape(np.rollaxis(X_train, 1, 0), (n_contrasts, n_nodes * n_subjects)).T
    groups =  np.hstack([i * np.ones(n_nodes, int) for i in range(n_subjects)])
    dictionary, labels = balanced_clustering(
        features, groups, n_components=n_clusters, reg=reg, write_dir=write_dir)
    # estimate shift
    shift = features - dictionary[labels]
    # average it per group
    for group in np.unique(groups):
        for label in np.unique(labels):
            mask = (groups == group) * (labels == label) 
            shift[mask] = np.mean(shift[mask], 0)
    # apply shift
    features -= shift
    X = np.array([features[groups == group].T for group in np.unique(groups)])
    return X, dictionary


def transform_equalization(X_test, dictionary, reg=.1):
    """ Transformer for the mulitvariate eqaulization function"""
    n_subjects, n_contrasts, n_nodes = X_test.shape
    features = np.reshape(np.rollaxis(X_test, 1, 0), (n_contrasts, n_nodes * n_subjects)).T
    groups =  np.hstack([i * np.ones(n_nodes, int) for i in range(n_subjects)])
    labels = np.zeros_like(groups)
    for group in np.unique(groups):
        R, _ = balanced_assignment(
            X_test[group].T, dictionary, reg=reg)
        labels[groups == group] = np.argmax(R, 1)
    shift = features - dictionary[labels]
    for group in np.unique(groups):
        for label in np.unique(labels):
            mask = (groups == group) * (labels == label) 
            shift[mask] = np.mean(shift[mask], 0)
    # apply shift
    features -= shift
    X = np.array([features[groups == group].T for group in np.unique(groups)])
    return X


from sklearn.base import TransformerMixin, BaseEstimator
class MultivariateEqualizer(TransformerMixin, BaseEstimator):
    """ Class to perform Multivariate Equalization with an sklearn API
    """

    def __init__(self, n_groups, reg=.1, write_dir='/tmp'):
        self.n_groups=n_groups
        self.reg=reg
        self.write_dir = write_dir

    def fit(self, X, y=None):
        _, dictionary = multivariate_equalization(X, self.n_groups, self.reg, self.write_dir)
        self.dictionary = dictionary
    
    def fit_transform(self, X, y=None):
        X_, dictionary = multivariate_equalization(X, self.n_groups, self.reg, self.write_dir)
        self.dictionary = dictionary
        return X_

    def transform(self, X, y=None):
        X_ = transform_equalization(X, self.dictionary, self.reg)
        return X_

        
if __name__ == '__main__':
    pass
    # test_balanced_clustering()
