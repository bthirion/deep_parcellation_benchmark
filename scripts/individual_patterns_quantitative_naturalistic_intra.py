"""
Does the same and individual_patterns quantitative, but providing quantitative assessment instead.
In this script, we train on naturalistic stimuli and validate on contrasts
As opposed to ther scripts, we measure within-subject goodness of fit
"""
import nibabel as nib
import os
import pandas as pd
import numpy as np
from joblib import Memory, Parallel, delayed
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import (
    plot_surf_roi, view_surf, plot_surf_stat_map, plot_surf_contours)
from nilearn.surface import load_surf_mesh
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import SparseCoder

from ibc_public.utils_data import make_surf_db, CONTRASTS
from utils_surface_plots import make_surf_data
from utils_dictionary import MultivariateEqualizer, make_dictionary



if os.path.exists('/neurospin'):
    DERIVATIVES = '/neurospin/ibc/derivatives'
    cache = '/neurospin/tmp/bthirion'
else:
    DERIVATIVES = '/storage/store2/data/ibc/derivatives'
    cache = '/storage/store2/work/bthirion/'

n_jobs = 10
highres = True
n_templates = 100
n_components = 20
alpha = 2.

write_dir = os.path.join(cache, 'roi', 'dictionary')

if not os.path.exists(write_dir):
    os.mkdir(write_dir)

mem = Memory(location=write_dir, verbose=0)

###############################################################################
# Get and fetch data: T-fMRI

subjects = [# 'sub-01',
            'sub-04', 'sub-05', 'sub-06', 'sub-07',  'sub-08',
            'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15'] # don't have the data for sub-01
n_subjects = len(subjects)

# Access to the data
task_list = ['ArchiEmotional', 'ArchiSocial', 'ArchiSpatial',
             'ArchiStandard', 'HcpEmotion', 'HcpGambling',
             'HcpLanguage', 'HcpMotor', 'HcpRelational',
             'HcpSocial', 'HcpWm',
             'RSVPLanguage']
task_list += ['preference', 'PreferenceFaces', 'PreferenceHouses',
              'PreferenceFood', 'PreferencePaintings']
task_list += ['MCSE', 'Moto', 'Visu', 'Audi',
             'MVEB', 'MVIS', 'Lec1', 'Lec2',
             'Bang', 'wedge', 'ring']
task_list += ['MTTNS', 'MTTWE', 'retinotopy', 'TheoryOfMind', 'Audio',
              'PainMovie', 'EmotionalPain', 'Enumeration', 'VSTM', 'Self'] 


training_tasks = task_list[:11]
test_tasks = task_list # task_list[11:]

mesh = 'fsaverage5'
if highres:
    mesh = 'fsaverage7'
    
# pick the corresponding mesh from Nilearn
fsaverage = fetch_surf_fsaverage(mesh)

###############################################################################
# Getting the training data, manually atm
import glob

tasks = ['ClipsTrn', 'Raiders', 'LePetitPrince', 'GoodBadUgly', 'Bang']
training_data = []
subjects_data = []
contrast_data = []
hemi_data = []
side = 'rh'
for subject in subjects:
    for task in tasks:
        wc = os.path.join(DERIVATIVES, subject, 'ses-*',
                          'res_task-{}_space-{}_dir-ffx'.format(task, mesh),
                          'effect_size_maps',
                          'srm_components_{}_{}.gii'.format(task, side))
        func = glob.glob(wc)
        if len(func) != 1:
            stop

        training_data.append(func[0])
        subjects_data.append(subject)
        contrast_data.append(task)
        hemi_data.append(side)

# make a df out of that
# must have columns: contrast, subject, side
data_dict = {'path': training_data,
             'contrast': contrast_data,
             'subject': subjects_data,
             'side': hemi_data}
df_train = pd.DataFrame(data_dict)

###############################################################################
# Load data

df_test = make_surf_db(
    derivatives=DERIVATIVES, conditions=CONTRASTS,
    subject_list=subjects, task_list=test_tasks, mesh=mesh,
    acquisition='ffx'
)

X_train, contrasts_train = make_surf_data(
    df_train, subjects, do_equalization=False, do_standardize=True, hemi=side)

contrasts_train = np.array(contrasts_train)

X_test, contrasts_test = make_surf_data(
    df_test, subjects, do_equalization=False, do_standardize=True, hemi=side)
contrasts_test = np.array(contrasts_test)

################################################################################
# Region selection
from sklearn.metrics.pairwise import cosine_distances

def cosine_distance(X1, X2):
    dist = []
    ns = X1.shape[0]
    if len(X2.shape) == 2:
        for i in range(ns):
            dist.append(np.diag(cosine_distances(X1[i], X2)))
    else:
        for i in range(ns):
            dist.append(np.diag(cosine_distances(X1[i], X2[i])))
    return np.array(dist)


def compute_errors(src_index, trg_index, X_test_mask, X_train_mask,
                   alpha, n_components, n_templates, reg=.1, masked_atlas=None):
    """ Evaluate the different methods"""
    meuh = MultivariateEqualizer(n_templates, reg=reg, write_dir=write_dir)
    n_voxels = X_test_mask[0].shape[1]
    n_contrasts = X_test_mask[0].shape[0]

    # prediction from zero_padding
    error_zero = np.linalg.norm(X_test_mask[trg_index], axis=2)
    # error_zero = np.zeros(n_contrasts)
    
    # prediction of the left out contrast simply using the mean
    mean_test = X_test_mask[src_index].mean(0)
    error_mean = np.linalg.norm(X_test_mask[trg_index] - mean_test, axis=2)
    error_mean = cosine_distance(X_test_mask[trg_index], mean_test)
    
    # Re-train the model on train subjects
    X_src_train = np.array(X_train_mask[src_index])
    # X_src_train_eq = meuh.fit_transform(X_src_train) ### skip to save time ? 
    X_src_train_eq = X_src_train
    
    dictionary, src_components = make_dictionary(
        np.hstack(X_src_train_eq), n_components=n_components, alpha=alpha,
        write_dir=write_dir, contrasts=contrasts_train, method='online')
    
    X_trg_train = np.array(X_train_mask[trg_index])
    # X_trg_train_eq = meuh.transform(X_trg_train) ### skip to save time ? 
    X_trg_train_eq = X_trg_train
    
    coder = SparseCoder(dictionary=dictionary, transform_algorithm='lasso_lars',
                        transform_alpha=alpha)
    trg_components = coder.transform((np.hstack(X_trg_train_eq).T))
    n_targets = len(trg_index)
    individual_components = trg_components.reshape(
        n_targets, n_voxels, n_components)

    stacked_X_src  = np.vstack([x.T for x  in X_test_mask[src_index]])
    test_dictionary = np.linalg.pinv(src_components).dot(stacked_X_src)
    prediction = np.array([
        individual_components[i].dot(test_dictionary).T
        for i in range(n_targets)])
    error_dic = np.linalg.norm(X_test_mask[trg_index] - prediction, axis=2)
    error_dic = cosine_distance(X_test_mask[trg_index], prediction)
    
    # Now without histogram equalization
    dictionary, src_components = make_dictionary(
        np.hstack(X_src_train), n_components=n_components, alpha=alpha,
        write_dir=write_dir, contrasts=contrasts_train, method='online')
    coder = SparseCoder(dictionary=dictionary, transform_algorithm='lasso_lars',
                        transform_alpha=alpha)
    trg_components = coder.transform((np.hstack(X_trg_train).T))
    individual_components = trg_components.reshape(
        n_targets, n_voxels, n_components)
        
    test_dictionary = np.linalg.pinv(src_components).dot(stacked_X_src)
    prediction = np.array([
        individual_components[i].dot(test_dictionary).T
        for i in range(n_targets)])
    error_dic_only_ = np.linalg.norm(X_test_mask[trg_index] - prediction, axis=2)
    error_dic_only_ = cosine_distance(X_test_mask[trg_index], prediction)

    # dic on concatenated data
    dictionary, src_components = make_dictionary(
        X_src_train.mean(0), n_components=n_components, alpha=alpha,
        write_dir=write_dir, contrasts=contrasts_train, method='online')
    coder = SparseCoder(dictionary=dictionary, transform_algorithm='lasso_lars',
                        transform_alpha=alpha)
    trg_components = coder.transform(X_trg_train.mean(0).T)    
    test_dictionary = np.linalg.pinv(src_components).dot(
        X_test_mask[src_index].mean(0).T)
    prediction = trg_components.dot(test_dictionary).T
    error_concat_dic_ = np.linalg.norm(
        X_test_mask[trg_index] - prediction, axis=2)
    error_concat_dic_ = cosine_distance(X_test_mask[trg_index], prediction)

    # Glasser ROIs mean
    prediction = np.zeros_like(X_test_mask[trg_index])
    for i in np.unique(masked_atlas):
        pred = X_test_mask[src_index].mean(0)[:, masked_atlas == i].mean(1)
        for  j in range(n_contrasts):
            prediction[:, j, masked_atlas == i] = pred[j]
    error_roi_ = np.linalg.norm(
        X_test_mask[trg_index] - prediction, axis=2)
    error_roi_ = cosine_distance(X_test_mask[trg_index], prediction)

    return (error_dic.mean(),
            error_mean.mean(),
            error_dic_only_.mean(),
            error_zero.mean(),
            error_concat_dic_.mean(),
            error_roi_.mean()
    )


def run_experiment_intra(
        X_train_mask, X_test_mask, alphas, reg=.1,
        n_jobs=n_jobs, region_id='', hemi='lh', n_components=10,
        masked_atlas=None):
    """ test generalization within individuals"""
    for alpha in alphas:
        results = Parallel(n_jobs=1)(
            delayed(compute_errors)
            (src_index, trg_index , X_test_mask, X_train_mask,
             alpha, n_components, n_templates, reg, masked_atlas=masked_atlas)
            for src_index, trg_index in zip([range(n_subjects)], [range(n_subjects)]))
        error_dics, error_means, error_dics_only, error_zeros, error_concat_dic, error_rois = np.array(results).T

        error_means = np.array(error_means)
        error_dics = np.array(error_dics)
        error_dics_only = np.array(error_dics_only)
        error_zeros = np.array(error_zeros)
        error_concat_dic = np.array(error_concat_dic)
        error_rois = np.array(error_rois)
        
        print(
            'error_zero: ', error_zeros.mean(),
            'error means: ', error_means.mean(),
            'error_dics_only: ', error_dics_only.mean(),
            'error_rois:', error_rois.mean(),
            'error_concat_dic: ', error_concat_dic.mean()
        )

        plt.figure(figsize=(5, 4))
        plt.boxplot([error_means, error_dics_only, error_dics, error_rois,
                     error_concat_dic])
        plt.xticks(range(1, 6),
                   ['population mean',
                    'dictionary',
                    'dictionary +\n equalization',
                    'rois',
                    'fixed\n dictionary'
                   ])
        plt.ylabel('prediction error')
        plt.savefig(os.path.join(
            write_dir, 'reconstruction_error_%d_%d_%f.png') %
                    (n_components, n_templates, alpha))

        results = {'error_means': error_means,
                   'error_dics': error_dics,
                   'error_dics_only': error_dics_only,
                   'error_zeros': error_zeros,
                   'error_rois': error_rois,
                   'error_concat': error_concat_dic,
        }
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(
            write_dir, 'results_intra_{}_{}_{}_{}_{}_naturalistic.csv').format
                  (region_id, hemi, n_components, n_templates, alpha))


# rs = ShuffleSplit(n_splits=20, test_size=.5, random_state=0)
alphas = [1.] # [.5, 1., 2., 3.]
reg = .1

if highres:
    glasser = nib.freesurfer.read_annot('{}.HCP-MMP1.annot'.format(side))[0]
    glasser_labels = [
        x.decode('UTF-8')[2:] for x in
        nib.freesurfer.read_annot('{}.HCP-MMP1.annot'.format(side))[2]]
    atlas = glasser

gve = pd.read_csv('../high_level_analysis_scripts/van_essen.csv',
                  dtype='str')
systems = [val.split(',') for val in gve.labels.values]

systems_contrast = pd.read_csv(os.path.join(
    write_dir, 'intra_subject_consistency_%s.csv' % side), index_col=0)
systems_contrast = systems_contrast[
    [c for c in contrasts_test if c in systems_contrast.columns]]

for system in range(len(systems)):
    
    mask = np.sum([
        atlas == i for i in np.array(systems[system], dtype=int)], 0)\
             .astype('bool')
    masked_atlas = atlas[mask]
    threshold = np.minimum(
        .5, np.nanpercentile(systems_contrast.values[system], 95))
    print(system, threshold)
    good_contrasts = systems_contrast.columns[
        systems_contrast.values[system] > threshold]
    contrast_mask = np.array([c in good_contrasts for c in contrasts_test])
    X_train_mask = np.array([x[:, mask] for x in X_train])
    X_test_mask = np.array([x[contrast_mask][:, mask] for x in X_test])
    n_components = len(systems[system])
    run_experiment_intra(X_train_mask, X_test_mask, alphas, 
                         reg=reg, n_jobs=n_jobs, region_id=system, hemi=side,
                         masked_atlas=masked_atlas, n_components=n_components,
    )

plt.show()
