""" Various utilities for surface-based plotting of brain maps
"""
import numpy as np
import nibabel as nib
import os
from nilearn import plotting
from nilearn import datasets


def make_surf_data(df, subject_list, hemi='lh',
                   do_equalization=False, do_standardize=False):
    """ Grep data for one hemisphere"""
    from utils_dictionary import histogram_equalization
    from sklearn.preprocessing import StandardScaler
    # first define a set of valid contrasts
    if do_standardize:
        scaler = StandardScaler()
    n_subjects =  len(subject_list)
    contrasts = df.contrast.unique()
    valid_contrasts = []
    for contrast in contrasts:
        if len(df[df.contrast == contrast].subject.unique()) == n_subjects:
            valid_contrasts.append(contrast)

    X = []
    for subject in subject_list:
        paths = []
        for contrast in valid_contrasts:
            mask = (df.contrast == contrast).values *\
                   (df.subject == subject).values
            if len(df[mask]) == 0:
                print(subject, contrast)
            paths.append(df[mask][df.side == hemi].path.values[-1])
        x = []    
        for texture in list(paths):
            x_ = nib.load(texture).darrays[0].data
            if x_.ndim == 1:
                x.append(np.atleast_2d(x_).T)
            elif x_.ndim == 2:
                x.append(x_)
        x = np.hstack(x).T
        x[np.isnan(x)] = 0
        if do_standardize:
            x = scaler.fit_transform(x)
        X.append(x)

    if do_equalization:
        X = histogram_equalization(X)
        
    X = np.array(X)
    return X, valid_contrasts


def surface_one_sample(df, contrast, side):
    from scipy.stats import ttest_1samp, norm
    mask = (df.contrast.values == contrast) * (df.side.values == side)
    X = np.array([nib.load(texture).darrays[0].data
                  for texture in list(df.path[mask].values)])
    # print (X.shape, np.sum(np.isnan(X)))
    t_values, p_values = ttest_1samp(X, 0)
    p_values = .5 * (1 - (1 - p_values) * np.sign(t_values))
    z_values = norm.isf(p_values)
    return z_values


def surface_conjunction(df, contrast, side, percentile=25):
    from conjunction import _conjunction_inference_from_z_values
    mask = (df.contrast.values == contrast) * (df.side.values == side)
    Z = np.array([nib.load(texture).darrays[0].data
                  for texture in list(df.path[mask].values)]).T
    pos_conj = _conjunction_inference_from_z_values(Z, percentile * .01)
    neg_conj = _conjunction_inference_from_z_values(-Z, percentile * .01)
    conj = pos_conj
    conj[conj < 0] = 0
    conj[neg_conj > 0] = - neg_conj[neg_conj > 0]
    return conj


def make_thumbnail_surface(func, hemi, threshold=3.0, vmax=10.,
                           output_dir='/tmp'):
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage', data_dir='/tmp')
    if hemi == 'right':
        mesh = fsaverage['infl_right']
        bg_map = fsaverage['sulc_right']
    else:
        mesh = fsaverage['infl_left']
        bg_map = fsaverage['sulc_left']

    medial = '/tmp/surf_medial_%s.png' % hemi
    lateral = '/tmp/surf_lateral_%s.png' % hemi
    # threshold = fdr_threshold(func, .05)
    plotting.plot_surf_stat_map(mesh, func, hemi=hemi, vmax=vmax,
                                threshold=threshold, bg_map=bg_map,
                                view='lateral', output_file=lateral)
    plotting.plot_surf_stat_map(mesh, func, hemi=hemi, vmax=vmax,
                                threshold=threshold, bg_map=bg_map,
                                view='medial', output_file=medial)
    return medial, lateral


def make_atlas_surface(label, hemi, name='', output_dir='/tmp'):
    """ Plot an atlas on the cortical surface """
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage', data_dir='/tmp')
    if hemi == 'right':
        mesh = fsaverage['infl_right']
        bg_map = fsaverage['sulc_right']
    else:
        mesh = fsaverage['infl_left']
        bg_map = fsaverage['sulc_left']

    medial = os.path.join(output_dir, '%s_medial_%s.png' % (name, hemi))
    lateral = os.path.join(output_dir, '%s_lateral_%s.png' % (name, hemi))
    plotting.plot_surf_roi(mesh, label, hemi=hemi, bg_map=bg_map,
                           view='lateral', output_file=lateral, alpha=.9)
    plotting.plot_surf_roi(mesh, label, hemi=hemi, bg_map=bg_map,
                           view='medial', output_file=medial, alpha=.9)


def faces_2_connectivity(faces):
    from scipy.sparse import coo_matrix
    n_features = len(np.unique(faces))
    edges = np.vstack((faces.T[:2].T, faces.T[1:].T, faces.T[0:3:2].T))
    weight = np.ones(edges.shape[0])
    connectivity = coo_matrix((weight, (edges.T[0], edges.T[1])),
                              (n_features, n_features))  # .tocsr()
    # Making it symmetrical
    connectivity = (connectivity + connectivity.T) / 2
    return connectivity


def connected_components_cleaning(connectivity, _map, cluster_size=10):
    from scipy.sparse import csgraph, coo_matrix
    n_features = connectivity.shape[0]
    weight = connectivity.data.copy()
    edges = connectivity.nonzero()
    i_idx, j_idx = edges
    weight[_map[i_idx] == 0] = 0
    weight[_map[j_idx] == 0] = 0
    mask = weight != 0
    reduced_connectivity = coo_matrix(
        (weight[mask], (i_idx[mask], j_idx[mask])), (n_features, n_features))
    # Clustering step: getting the connected components of the nn matrix
    n_components, labels = csgraph.connected_components(reduced_connectivity)
    label, count = np.unique(labels, return_counts=True)
    good_labels = label[count >= cluster_size]
    map_ = np.zeros_like(_map)
    for gl in good_labels:
        map_[labels == gl] = _map[labels == gl]
    return map_


def clean_surface_map(maps, hemi, cluster_size):
    """Clean surface maps by removing small connected components"""
    from nilearn.surface import load_surf_mesh
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage', data_dir='/tmp')
    if hemi == 'right':
        mesh = fsaverage['infl_right']
    else:
        mesh = fsaverage['infl_left']
    
    _, faces = load_surf_mesh(mesh)
    connectivity = faces_2_connectivity(faces)
    for i in range(maps.shape[1]):
        maps[:, i] = connected_components_cleaning(
            connectivity, maps[:, i], cluster_size=cluster_size)
    return maps
    

def map_surface_label(components, name, output_dir, facecolor='k'):
    from utils_surface_plots import make_atlas_surface
    from nibabel.gifti import GiftiDataArray, GiftiImage
    mask = components.max(1) > 0
    n_voxels = components.shape[0] // 2
    labels = np.zeros(components.shape[0]).astype(np.int)
    labels[mask] = np.argmax(components, 1)[mask] + 1
    make_atlas_surface(
        labels[:n_voxels], 'left', name, output_dir)
    make_atlas_surface(
        labels[n_voxels:], 'right', name, output_dir)
    GiftiImage(
        darrays=[GiftiDataArray().from_array(
            labels[:n_voxels], 'NIFTI_INTENT_ESTIMATE')]
    ).to_filename(os.path.join(output_dir, '%s_lh.gii' % name))
    GiftiImage(
        darrays=[GiftiDataArray().from_array(
            labels[n_voxels:], 'NIFTI_INTENT_ESTIMATE')]
    ).to_filename(os.path.join(output_dir, '%s_rh.gii' % name))


def mesh_to_graph(mesh):
    """Convert a mesh into a connectivity matrix

    Parameters
    ----------
    mesh: path to a mesh file or mesh file

    Returns
    -------
    connectivity: sparse matrix represneting the mesh

    """
    from scipy.sparse import coo_matrix
    from nilearn.surface import load_surf_mesh
    try:
        coords, triangles = load_surf_mesh(mesh)
        # coords, triangles = nib.freesurfer.read_geometry(mesh)
    except:
        from nibabel.gifti import read
        arr = read(mesh).darrays
        coords = arr[0].data
        triangles = arr[1].data
    n_points = coords.shape[0]
    edges = np.hstack((
        np.vstack((triangles[:, 0], triangles[:, 1])),
        np.vstack((triangles[:, 0], triangles[:, 2])),
        np.vstack((triangles[:, 1], triangles[:, 0])),
        np.vstack((triangles[:, 1], triangles[:, 2])),
        np.vstack((triangles[:, 2], triangles[:, 0])),
        np.vstack((triangles[:, 2], triangles[:, 1])),
        ))
    weights = np.ones(edges.shape[1])
    
    connectivity = coo_matrix((weights, edges), (n_points, n_points)).tocsr()

    # Making it symmetrical
    connectivity = (connectivity + connectivity.T) / 2

    return connectivity


