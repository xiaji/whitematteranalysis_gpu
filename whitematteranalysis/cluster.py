# -*- coding: utf-8 -*-

""" cluster.py

implementations of fiber clustering

"""
import os
import pickle

import vtk
import numpy
try:
    import scipy.cluster.vq
    import scipy.cluster.hierarchy
    USE_SCIPY = 1
except ImportError:
    USE_SCIPY = 0
    print(f"<{os.path.basename(__file__)}> Failed to import scipy.cluster, cannot cluster.")
    print(f"<{os.path.basename(__file__)}> Please install scipy for this functionality.")
try:
    from joblib import Parallel, delayed
    USE_PARALLEL = 1
except ImportError:
    USE_PARALLEL = 0
    print(f"<{os.path.basename(__file__)}> Failed to import joblib, cannot multiprocess.")
    print(f"<{os.path.basename(__file__)}> Please install joblib for this functionality.")

from . import fibers
from . import similarity
from . import filter
from . import render
from . import io
from . import mrml
from . import gpu

from pprint import pprint

HAVE_PLT = 1
try:
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except:
    print(f"<{os.path.basename(__file__)}> Error importing matplotlib.pyplot package, can't plot quality control data.\n")
    HAVE_PLT = 0
    
# This did not work better. Leave here for future testing if of interest
if 0:
    try:    
        from sklearn.cluster import AffinityPropagation
        from sklearn import metrics
    except ImportError:
        SKLEARN = 0
        print(f"<{os.path.basename(__file__)}> Failed to import sklearn, cannot use affinity propagation.")
        print(f"<{os.path.basename(__file__)}> Please install sklearn for this functionality.")
    
class ClusterAtlas:
    """Variables necessary to label a new subject from a spectral cluster atlas."""

    def __init__(self):
        self.pinv_A = None
        self.e_val = None
        self.e_vec = None
        self.row_sum_matrix = None
        self.row_sum_1 = None
        self.e_vec_norm = None
        self.nystrom_polydata = None
        self.nystrom_coords = None
        self.nystrom_count = None
        self.threshold = None
        self.number_of_eigenvectors = None
        self.centroids = None
        self.sigma = None
        self.bilateral = None
        self.distance_method = None
        self.version = 1.0
        self.polydata_filename = None
        # Outlier removal
        self.cluster_outlier_std_threshold = None
        self.cluster_mean_similarity = None
        self.cluster_std_similarity = None
        self.brain_mean_similarity = None
        self.brain_std_similarity = None
        
    def save(self, directory, atlas_name):
        # temporarily remove polydata object to enable pickling
        polydata_tmp = self.nystrom_polydata
        coords_tmp = getattr(self, 'nystrom_coords', None)
        self.nystrom_polydata = None
        self.nystrom_coords = None
        fname = os.path.join(directory,atlas_name)
        with open(fname + '.p', 'wb') as atlas_file:
            pickle.dump(self, atlas_file)
        # save the polydata
        io.write_polydata(polydata_tmp, fname+'.vtp')
        # replace the polydata object
        self.nystrom_polydata = polydata_tmp
        self.nystrom_coords = coords_tmp

    def load(self, directory, atlas_name, verbose=False):
        if not os.path.isdir(directory):
            print("Error: Atlas directory", directory, "does not exist or is not a directory.")
            raise "<cluster.py> I/O error"
        
        fname_base = os.path.join(directory,atlas_name)
        fname_atlas = fname_base+'.p'
        fname_polydata = fname_base+'.vtp'
        
        if not os.path.exists(fname_atlas):
            print("Error: Atlas file", fname_atlas, "does not exist.")
            raise "<cluster.py> I/O error"
        if not os.path.exists(fname_polydata):
            print("Error: Atlas file", fname_polydata, "does not exist.")
            raise "<cluster.py> I/O error"

        try:
            with open(fname_atlas, 'rb') as atlas_file:
                atlas = pickle.load(atlas_file)
        except UnicodeDecodeError:
            with open(fname_atlas, 'rb') as atlas_file:
                atlas = pickle.load(atlas_file, encoding="latin1")
        if not isinstance(getattr(atlas, 'bilateral', None), (bool, numpy.bool_)):
            raise ValueError("Atlas must specify a Boolean bilateral field; rebuild the atlas with wm_cluster_atlas.py.")
        atlas.nystrom_polydata = io.read_polydata(fname_polydata)
        print(f"<{os.path.basename(__file__)}> Loaded atlas", atlas_name, "from", directory, ".")
        print(f"<{os.path.basename(__file__)}> Atlas Nystrom polydata sample:", atlas.nystrom_polydata.GetNumberOfLines(), \
            "\nAtlas size:", atlas.pinv_A.shape, "\nAtlas number of eigenvectors:", atlas.number_of_eigenvectors)

        atlas.polydata_filename = fname_polydata

        # check for any version issues
        if not hasattr(atlas, 'version'):    
            atlas.version = '0.0'
        # this is an atlas from pre 1.0 version code. Set to default values.
        if not hasattr(atlas, 'distance_method'):    
            atlas.distance_method = 'Hausdorff'

        if verbose:
            pprint (vars(atlas))

        return(atlas)

def load_atlas(directory, atlas_name, for_labeling=False):
    """ Convenience function to load a spectral cluster atlas """
    if for_labeling:
        return gpu.atlas_cache.load_spectral_atlas(directory, atlas_name)
    tmp_atlas = ClusterAtlas()
    return tmp_atlas.load(directory, atlas_name)


def _atlas_nystrom_count(atlas):
    coords = getattr(atlas, 'nystrom_coords', None)
    if coords is not None:
        return int(len(coords))
    if getattr(atlas, 'nystrom_polydata', None) is not None:
        return int(atlas.nystrom_polydata.GetNumberOfLines())
    if getattr(atlas, 'nystrom_count', None) is not None:
        return int(atlas.nystrom_count)
    return int(len(atlas.row_sum_1))


def _ensure_nystrom_polydata(atlas):
    if getattr(atlas, 'nystrom_polydata', None) is None:
        fname = getattr(atlas, 'polydata_filename', None)
        if not fname or not os.path.exists(fname):
            raise RuntimeError("Atlas is missing nystrom_polydata and polydata_filename")
        atlas.nystrom_polydata = io.read_polydata(fname)
    return atlas.nystrom_polydata


def _atlas_nystrom_coords(atlas, coordinate_major=False):
    coords = getattr(atlas, 'nystrom_coords', None)
    if coords is not None:
        coords = numpy.asarray(coords, dtype=numpy.float64)
        if coordinate_major:
            return numpy.ascontiguousarray(coords.transpose(2, 0, 1)).transpose(1, 2, 0)
        return coords
    pd = _ensure_nystrom_polydata(atlas)
    coords = gpu.fiber_tensors.polydata_to_numpy(pd, 15, coordinate_major=coordinate_major)
    if not coordinate_major:
        atlas.nystrom_coords = coords
    return coords
        
def hierarchical(input_polydata, number_of_clusters=300,
                 threshold=2, fcluster_threshold=0.4,
                 number_of_jobs=3):
    """ This is just a test. This is only a test.  Stay tuned."""

    distance_matrix = \
        _pairwise_distance_matrix(input_polydata, threshold,
                                  number_of_jobs)

    # convert to upper triangular [not needed]
    # (note should compute only this part.)
    #mask_matrix = numpy.ones(distance_matrix.shape)
    #mask_matrix = numpy.triu(mask_matrix)

    #distance_matrix_triu = distance_matrix(numpy.nonzero(mask_matrix))

    z_link = scipy.cluster.hierarchy.linkage(distance_matrix)
    cluster_idx = scipy.cluster.hierarchy.fcluster(z_link, fcluster_threshold)

    cluster_colors = vtk.vtkIntArray()
    cluster_colors.SetName('ClusterNumber')
    output_polydata = input_polydata
    for lidx in range(0, output_polydata.GetNumberOfLines()):
        cluster_colors.InsertNextTuple1(cluster_idx[lidx])

    output_polydata.GetCellData().AddArray(cluster_colors)
    output_polydata.GetCellData().SetActiveScalars('ClusterNumber')

    return output_polydata, cluster_idx


def nearPSD(A, epsilon=0, eig=None):
    """Project a symmetric matrix to the nearest correlation-style PSD matrix via eigh."""
    if eig is not None:
        eigval, eigvec = eig
    else:
        eigval, eigvec = gpu.linalg.eigh(A)
    val = numpy.maximum(eigval, epsilon)
    denom = (eigvec * eigvec) @ val
    inv_diag_sqrt = numpy.sqrt(1.0 / denom)
    factors = (inv_diag_sqrt[:, None] * eigvec) * numpy.sqrt(val)[None, :]
    return factors @ factors.T
   
def spectral(input_polydata, number_of_clusters=200,
             number_of_eigenvectors=20, sigma=60, threshold=0.0,
             number_of_jobs=3, use_nystrom=False, nystrom_mask=None,
             landmarks=None, distance_method='Mean', normalized_cuts=True,
             outlier_std_threshold = 2.0,
             pos_def_approx=True,
             bilateral=False,
             use_gpu=None, gpu_float32=None):

    """ Spectral clustering based on pairwise fiber affinity matrix.

    As in O'Donnell and Westin TMI 2007.

    Differences from that implementation: fiber point correspondences are defined
    using fixed-length fiber parameterization (instead of closest point).

    """

    # test pd has lines first
    number_fibers = input_polydata.GetNumberOfLines()
    print(f"<{os.path.basename(__file__)}> Starting spectral clustering.")
    print(f"<{os.path.basename(__file__)}> Number of input fibers:", number_fibers)
    print(f"<{os.path.basename(__file__)}> Number of clusters:", number_of_clusters)

    if number_fibers == 0:
        print(f"<{os.path.basename(__file__)}> ERROR: Cannot cluster polydata with 0 fibers.")
        return

    atlas = ClusterAtlas() 

    # Store all parameters to this function. They must be identical later to label new data.
    # Below, calculated values will also be stored in the atlas.
    atlas.number_of_eigenvectors = number_of_eigenvectors
    atlas.sigma = sigma
    atlas.threshold = threshold
    atlas.use_nystrom = use_nystrom
    atlas.landmarks = landmarks
    atlas.distance_method = distance_method
    atlas.bilateral = bilateral

    # 1) Compute fiber similarities.
    # Nystrom version of the code uses a sample of the data.
    if use_nystrom:
        # make sure it's an array for logic operations
        nystrom_mask = numpy.array(nystrom_mask)
        # make sure it's boolean or 0 and 1
        test = numpy.max(nystrom_mask) == 1.0
        if not test:
            print(f"<{os.path.basename(__file__)}> ERROR: Nystrom data mask is may not be Boolean. Max value is not 1.0/True.")
            raise AssertionError
        # make sure it's large enough
        test = sum(nystrom_mask) >= 100
        if not test:
            print(f"<{os.path.basename(__file__)}> ERROR: Nystrom data mask is smaller than 100.")
            raise AssertionError
        # make sure its size matches the polydata input
        test = len(nystrom_mask) == number_fibers
        if not test:
            print(f"<{os.path.basename(__file__)}> ERROR: Nystrom data mask size does not match polydata number of lines.")
            raise AssertionError
        
        # Separate the Nystrom sample and the rest of the data.
        polydata_m = filter.mask(input_polydata, nystrom_mask, verbose=False)
        atlas.nystrom_polydata = polydata_m
        polydata_n = filter.mask(input_polydata, nystrom_mask == False, verbose=False)
        sz = polydata_m.GetNumberOfLines()
        print(f'<{os.path.basename(__file__)}> Using Nystrom approximation. Subset size:',  sz, '/', number_fibers)
        # Determine ordering to get embedding to correspond to original input data.
        reorder_embedding = numpy.concatenate((numpy.where(nystrom_mask)[0], numpy.where(~nystrom_mask)[0]))
        if landmarks is not None:
            landmarks_m = landmarks[nystrom_mask,:,:]
            landmarks_n = landmarks[~nystrom_mask,:,:]
        else:
            landmarks_m = landmarks_n = None

        # Calculate fiber similarities
        A = \
            _pairwise_similarity_matrix(polydata_m, threshold,
                                        sigma, number_of_jobs, landmarks_m, distance_method, bilateral,
                                        use_gpu=use_gpu, gpu_float32=gpu_float32)
        B = \
            _rectangular_similarity_matrix(polydata_n, polydata_m, threshold,
                                           sigma, number_of_jobs, landmarks_n, landmarks_m, distance_method, bilateral,
                                           use_gpu=use_gpu, gpu_float32=gpu_float32)

        # sanity check
        print(f"<{os.path.basename(__file__)}> Range of values in A:", numpy.min(A), numpy.max(A))
        print(f"<{os.path.basename(__file__)}> Range of values in B:", numpy.min(B), numpy.max(B))
        
    else:
        # Calculate all fiber similarities
        A = \
            _pairwise_similarity_matrix(input_polydata, threshold,
                                    sigma, number_of_jobs, landmarks, distance_method, bilateral,
                                    use_gpu=use_gpu, gpu_float32=gpu_float32)

        atlas.nystrom_polydata = input_polydata
        # sanity check
        print(f"<{os.path.basename(__file__)}> Range of values in A:", numpy.min(A), numpy.max(A))
        
    testval = numpy.max(A-A.T) 
    if not testval == 0.0:
        if testval > 1e-10:
            print(f"<{os.path.basename(__file__)}> ERROR: A matrix is not symmetric.")
            raise AssertionError
        else:
            print(f"<{os.path.basename(__file__)}> Maximum of A - A^T:", testval)
        # Ensure that A is symmetric
        A = numpy.divide(A+A.T, 2.0)
        
    testval = numpy.min(A)
    if not testval > 0.0:
        print(f"<{os.path.basename(__file__)}> ERROR: A matrix is not positive.")
        print(f"<{os.path.basename(__file__)}> Minimum value in A: ", testval)
        if testval < 0.0:
            raise AssertionError

    # Outliers will have low measured (or estimated) row sums. Detect outliers in A:
    # to turn off for testing: outlier_std_threshold = numpy.inf
    row_sum_A_initial = numpy.sum(A, axis=0) + numpy.sum(B.T, axis=0)
    print(f"<{os.path.basename(__file__)}> Initial similarity (row) sum A:", numpy.mean(row_sum_A_initial), numpy.std(row_sum_A_initial), numpy.min(row_sum_A_initial))
    atlas.outlier_std_threshold = outlier_std_threshold
    atlas.row_sum_threshold_for_rejection = numpy.mean(row_sum_A_initial) - outlier_std_threshold*numpy.std(row_sum_A_initial)
    bad_idx = numpy.nonzero(row_sum_A_initial < atlas.row_sum_threshold_for_rejection)[0]
    reject_A = bad_idx
    print(f"<{os.path.basename(__file__)}> Rejecting n=", len(bad_idx), "/", sz, "fibers >", outlier_std_threshold, "standard deviations below the mean total fiber similarity")

    A = numpy.delete(A,reject_A,0)
    A = numpy.delete(A,reject_A,1)
    #print A.shape, B.shape
    B = numpy.delete(B,reject_A,0)
    #print A.shape, B.shape, reorder_embedding.shape
                    
    # Ensure that A is positive definite.
    e_val_a = None
    e_vec_a = None
    if pos_def_approx:
        e_val, e_vec = gpu.linalg.eigh(A)
        print(f"<{os.path.basename(__file__)}> Eigenvalue range of A:", e_val[0], e_val[-1])
        psd_tol = 1e-12 * max(1.0, abs(float(e_val[-1]))) if e_val.size else 0.0
        if e_val.size > 0 and e_val[0] >= -psd_tol and numpy.allclose(numpy.diag(A), 1.0, atol=1e-12, rtol=0.0):
            e_val_a, e_vec_a = numpy.maximum(e_val, 0.0), e_vec
            print(f"<{os.path.basename(__file__)}> A is already PSD with unit diagonal; skipping nearPSD.")
        else:
            A2 = nearPSD(A, eig=(e_val, e_vec))
            e_val_a, e_vec_a = gpu.linalg.eigh(A2)
            print(f"<{os.path.basename(__file__)}> Eigenvalue range of nearest PSD matrix to A:", e_val_a[0], e_val_a[-1])
            testval = numpy.max(A-A2) 
            if not testval == 0.0:
                print(f"<{os.path.basename(__file__)}> A matrix differs by PSD matrix by maximum of:", testval)
                if testval > 0.25:
                    print(f"<{os.path.basename(__file__)}> ERROR: A matrix changed by more than 0.25.")
                    raise AssertionError
            A = A2
        
    # 2) Do Normalized Cuts transform of similarity matrix.
    # See the paper: "Spectral Grouping Using the Nystrom Method"
    # (D^-1/2 W D^-1/2) V = V Lambda
    if normalized_cuts:
        if use_nystrom:
            # Form of entire affinity matrix: 
            # A   B
            # B^T   C
            # C is not computed.
            # Calculate the sum of the partial rows we've computed:
            atlas.row_sum_1 = numpy.sum(A, axis=0) + numpy.sum(B.T, axis=0)  
            if e_val_a is not None:
                max_abs_ev = numpy.max(numpy.abs(e_val_a)) if e_val_a.size else 0.0
                cutoff = 1e-15 * max_abs_ev
                nonzero = numpy.abs(e_val_a) > cutoff
                inv_e_val = numpy.where(nonzero, numpy.divide(1.0, e_val_a, where=nonzero), 0.0)
                atlas.pinv_A = (e_vec_a * inv_e_val[None, :]) @ e_vec_a.T
            else:
                atlas.pinv_A = numpy.linalg.pinv(A)

            atlas.row_sum_matrix = numpy.dot(numpy.sum(B.T, axis=0), atlas.pinv_A)
            print(f"<{os.path.basename(__file__)}> Range of row sum weights:", numpy.min(atlas.row_sum_matrix), numpy.max(atlas.row_sum_matrix))
            
            # row sum estimate for current B part of the matrix
            if gpu.use_gpu(use_gpu):
                row_sum_2 = gpu.linalg.nystrom_row_sum_2(B, atlas.row_sum_matrix)
            else:
                row_sum_2 = numpy.sum(B, axis=0) + \
                    numpy.dot(atlas.row_sum_matrix, B)
            print(f"<{os.path.basename(__file__)}> Row sum check (min/max, should be > 0) A:", numpy.min(atlas.row_sum_1), numpy.median(atlas.row_sum_1), numpy.max(atlas.row_sum_1),  "B:", numpy.min(row_sum_2), numpy.median(row_sum_2), numpy.max(row_sum_2))

            # reject outliers in B
            bad_idx = numpy.nonzero(row_sum_2 < atlas.row_sum_threshold_for_rejection)[0]
            reject_B = bad_idx
            print(f"<{os.path.basename(__file__)}> Rejecting n=", len(bad_idx), "/", B.shape[1], "fibers >", outlier_std_threshold, "standard deviations below the mean total fiber similarity")
            row_sum_2 = numpy.delete(row_sum_2,reject_B)
            B = numpy.delete(B,reject_B,1)

            print(f"<{os.path.basename(__file__)}> After outlier rejection A:", A.shape, "B:", B.shape)
            print(f"<{os.path.basename(__file__)}> Row sum check (min/max, should be > 0) A:", numpy.min(atlas.row_sum_1), numpy.median(atlas.row_sum_1), numpy.max(atlas.row_sum_1),  "B:", numpy.min(row_sum_2), numpy.median(row_sum_2), numpy.max(row_sum_2))

            # Separate the Nystrom sample and the rest of the data after removing outliers
            nystrom_mask_2 = nystrom_mask
            midxA = numpy.nonzero(nystrom_mask_2)[0]
            nystrom_mask_2[midxA[reject_A]] = False
            not_nystrom_mask = nystrom_mask == False
            not_nystrom_mask[midxA[reject_A]] = False
            midxB = numpy.nonzero(not_nystrom_mask)[0]
            not_nystrom_mask[midxB[reject_B]] = False

            polydata_m = filter.mask(input_polydata, nystrom_mask_2, verbose=False)
            atlas.nystrom_polydata = polydata_m
            polydata_n = filter.mask(input_polydata, not_nystrom_mask, verbose=False)
            output_polydata = filter.mask(input_polydata, numpy.add(nystrom_mask_2, not_nystrom_mask),verbose=False)
            sz = polydata_m.GetNumberOfLines()
            number_fibers = output_polydata.GetNumberOfLines()
            print(f'<{os.path.basename(__file__)}> Using Nystrom approximation. Subset size (A):',  sz, '/', number_fibers, "B:", polydata_n.GetNumberOfLines())
            reject_idx = numpy.concatenate((midxA[reject_A],midxB[reject_B]))
            nystrom_mask_2 = numpy.delete(nystrom_mask_2,reject_idx)
            not_nystrom_mask = numpy.delete(not_nystrom_mask,reject_idx)
            reorder_embedding = numpy.concatenate((numpy.where(nystrom_mask_2)[0], numpy.where(not_nystrom_mask)[0]))
           
            # in case of negative row sum estimation
            if any(row_sum_2<=0):
                print(f"<{os.path.basename(__file__)}> Warning: Consider increasing sigma or using the Mean distance. negative row sum approximations.")
                print("Number of negative row sums:", numpy.count_nonzero(row_sum_2<=0))

            # save for testing (before normalization)
            column_sum = numpy.concatenate((numpy.sum(A, axis=1), numpy.sum(B.T, axis=1)))

            # normalized cuts normalization
            row_sum = numpy.concatenate((atlas.row_sum_1, row_sum_2))
            dhat = numpy.sqrt(numpy.divide(1, row_sum))
            dhat_a = dhat[0:sz]
            dhat_b = dhat[sz:]

            A = numpy.multiply(A, numpy.outer(dhat_a, dhat_a.T))
            if not gpu.use_gpu(use_gpu):
                B = numpy.multiply(B, numpy.outer(dhat_a, dhat_b.T))

        else:
            # normalized cuts normalization using row (same as column) sums
            row_sum = numpy.sum(A, axis=0)
            print(f"<{os.path.basename(__file__)}> A matrix row sums range (should be > 0):", numpy.min(row_sum), numpy.max(row_sum))
            dhat = numpy.divide(1, numpy.sqrt(row_sum))
            A = \
                numpy.multiply(A, numpy.outer(dhat, dhat.T))

    # 3) Compute eigenvectors for use in spectral embedding
    print(f'<{os.path.basename(__file__)}> Calculating eigenvectors of similarity matrix A...')
    atlas.e_val, atlas.e_vec = gpu.linalg.eigh(A)
    print(f'<{os.path.basename(__file__)}> Done calculating eigenvectors.')
    print(f"<{os.path.basename(__file__)}> Eigenvalue range:", atlas.e_val[0], atlas.e_val[-1])
    # Check how well our chosen number of eigenvectors models the data
    power = numpy.cumsum(atlas.e_val[::-1]) / numpy.sum(atlas.e_val)
    print(f"<{os.path.basename(__file__)}> Power from chosen number of eigenvectors (", number_of_eigenvectors, ')', power[number_of_eigenvectors])
    print(f'<{os.path.basename(__file__)}> Top eigenvalues:', atlas.e_val[::-1][1:number_of_eigenvectors])

    # 4) Compute embedding using eigenvectors
    print(f'<{os.path.basename(__file__)}> Compute embedding using eigenvectors.')
    if use_nystrom:
        count = min(number_of_eigenvectors + 1, atlas.e_vec.shape[1])
        e_vec_top = atlas.e_vec[:, -count:]
        e_val_top = atlas.e_val[-count:]
        if gpu.use_gpu(use_gpu):
            V2_top, norm_top = gpu.linalg.nystrom_normalize_and_embed_top(
                B, dhat_a, dhat_b, e_vec_top, e_val_top, gpu_float32=gpu_float32)
            V_top = numpy.concatenate((e_vec_top, V2_top))
        else:
            V2_top = numpy.dot(
                numpy.dot(B.T, e_vec_top),
                numpy.diag(numpy.divide(1.0, e_val_top)))
            V_top = numpy.concatenate((e_vec_top, V2_top))
            norm_top = numpy.sum(numpy.multiply(V_top, V_top), 0)

        atlas.e_vec_norm = numpy.ones(atlas.e_vec.shape[1], dtype=numpy.float64)
        atlas.e_vec_norm[-count:] = norm_top
        V_top = numpy.divide(V_top, norm_top)

        embed = numpy.zeros((number_fibers, number_of_eigenvectors))
        for i in range(0, number_of_eigenvectors):
            embed[reorder_embedding, i] = numpy.divide(V_top[:, -(i + 2)], V_top[:, -1])
    else:
        embed = atlas.e_vec[:, -number_of_eigenvectors - 2: -2]
        embed = numpy.divide(embed.T, atlas.e_vec[:, -1]).T
        # reverse order of embedding so highest eigenvalue
        # information is first
        embed = embed[:, ::-1]


    # Default is always k-means. Other code is just left for testing. Did not improve results.
    #centroid_finder = 'AffinityPropagation'
    centroid_finder = 'K-means'
    
    # 5) Find clusters using k-means in embedding space.
    cluster_metric = None
    if centroid_finder == 'K-means':
        print(f'<{os.path.basename(__file__)}> K-means clustering in embedding space.')
        if gpu.use_gpu(use_gpu) and number_of_clusters <= embed.shape[0]:
            centroids, cluster_metric = gpu.linalg.kmeans2(
                embed, number_of_clusters, n_iter=10, gpu_float32=gpu_float32)
        else:
            centroids, cluster_metric = scipy.cluster.vq.kmeans2(embed, number_of_clusters, minit='points')
        # sort centroids by first eigenvector order
        # centroid_order = numpy.argsort(centroids[:,0])
        # sort centroids according to colormap and save them in this order in atlas
        color = _embed_to_rgb(centroids)
        centroid_order = render.argsort_by_jet_lookup_table(color)
        atlas.centroids = centroids[centroid_order,:]
        cluster_idx, dist = scipy.cluster.vq.vq(embed, atlas.centroids)
        #print "<cluster.py> Distortion metric:", cluster_metric
        if 0:
            # This is extremely slow, but leave code here if ever wanted for testing
            cluster_metric = metrics.silhouette_score(embed, cluster_idx, metric='sqeuclidean')
            print("Silhouette Coefficient: %0.3f" % cluster_metric)
 
    else:
        print("ERROR: Unknown centroid finder", centroid_finder)
        ## # This found fewer clusters than we need to represent the anatomy well
        ## # Leave code here in case wanted in future for more testing.
        ## print '<cluster.py> Affinity Propagation clustering in embedding space.'
        ## af = AffinityPropagation(preference=-50).fit(embed)
        ## cluster_centers_indices = af.cluster_centers_indices_
        ## labels = af.labels_
        ## n_clusters_ = len(cluster_centers_indices)
        ## print('Estimated number of clusters: %d' % n_clusters_)
        ## cluster_idx = labels
        ## for k in range(n_clusters_):
        ##     class_members = labels == k
        ##     atlas.centroids = embed[cluster_centers_indices[k]]
        ## # return metrics
        ## if 0:
        ##     # This is extremely slow, but leave code here if ever wanted for testing
        ##     cluster_metric = metrics.silhouette_score(embed, labels, metric='sqeuclidean')
        ##     print("Silhouette Coefficient: %0.3f" % cluster_metric)

    # 6) Output results.
    print(f'<{os.path.basename(__file__)}> Done spectral clustering, returning results.')
    # visualize embedding coordinates as RGB
    embed2 = embed
    #embed2[numpy.isnan(embed)] = 0.0
    color = _embed_to_rgb(embed2)
    # set up polydata with clustering output info.
    # for now modify input polydata by adding two arrays
    #output_polydata = input_polydata
    if use_nystrom:
        output_polydata = \
            _format_output_polydata(output_polydata, cluster_idx, color, embed, row_sum[reorder_embedding], column_sum[reorder_embedding])
    else:
        # row and column sums are the same. no need to reorder.
        output_polydata = \
            _format_output_polydata(output_polydata, cluster_idx, color, embed, row_sum, row_sum)
            
    return output_polydata, cluster_idx, color, embed, cluster_metric, atlas, reject_idx 


def view_cluster_number(input_polydata, cluster_number, cluster_indices=None):

    """ Pop up a render window showing the selected cluster.


    Uses cluster_indices to choose corresponding cells in the
    polydata. If no argument cluster_indices is provided, then uses
    the cell data array named ClusterNumber.  One of these inputs must
    be present.

    """

    if cluster_indices == None:
        cluster_indices_vtk = \
            input_polydata.GetCellData().GetArray('ClusterNumber')
        cluster_indices = numpy.zeros(cluster_indices_vtk.GetNumberOfTuples())
        for fidx in range(0, cluster_indices_vtk.GetNumberOfTuples()):
            cluster_indices[fidx] = cluster_indices_vtk.GetTuple(fidx)[0]

    fiber_mask = cluster_indices == cluster_number
    view_polydata = filter.mask(input_polydata, fiber_mask)
    ren = render.render(view_polydata)

    return ren

def _lookup_prepared_gpu(prepared_gpu, dev):
    if prepared_gpu is None:
        return None
    if dev in prepared_gpu:
        return prepared_gpu[dev]
    key = (dev.type, dev.index)
    if key in prepared_gpu:
        return prepared_gpu[key]
    return None


def spectral_atlas_label(input_polydata, atlas, number_of_jobs=2,
                         use_gpu=None, gpu_float32=None,
                         queries_15=None, prepared_gpu=None):
    """ Use an existing atlas to label a new polydata.

    Returns the cluster indices for all the fibers. output_polydata, cluster_numbers, color, embed = wma.cluster.spectral_atlas_label(input_data, atlas)

    """

    number_fibers = input_polydata.GetNumberOfLines()
    sz = _atlas_nystrom_count(atlas)

    if gpu.use_gpu(use_gpu):
        n_gpus = gpu.visible_device_count()
        base_dev = gpu.get_device()
        queries = (numpy.asarray(queries_15, dtype=numpy.float64)
                   if queries_15 is not None
                   else gpu.fiber_tensors.polydata_to_numpy(input_polydata, 15))
        if n_gpus > 1 and number_fibers >= 16384 and base_dev.type != 'mps':
            from concurrent.futures import ThreadPoolExecutor
            import contextlib
            import torch

            print(f"<{os.path.basename(__file__)}> Computing rectangular similarity matrix across {n_gpus} GPUs (fused embedding).")
            fixed_host = None if prepared_gpu is not None else _atlas_nystrom_coords(atlas)
            bounds = numpy.linspace(0, number_fibers, n_gpus + 1, dtype=numpy.int64)
            shards = [(i, int(bounds[i]), int(bounds[i + 1]))
                      for i in range(n_gpus) if bounds[i + 1] > bounds[i]]

            def _label_shard(item):
                gpu_idx, start, end = item
                dev = gpu.get_device(gpu_idx)
                ctx = (torch.cuda.device(dev)
                       if dev.type == 'cuda' and torch.cuda.is_available()
                       else contextlib.nullcontext())
                with ctx:
                    entry = _lookup_prepared_gpu(prepared_gpu, dev)
                    if entry is not None:
                        fixed_arg, prepared_arg = entry
                    else:
                        fixed_arg = fixed_host if fixed_host is not None else _atlas_nystrom_coords(atlas)
                        prepared_arg = None
                    blocks = gpu.distance._iter_rectangular_similarity_arrays(
                        queries[start:end], fixed_arg,
                        atlas.threshold, atlas.sigma,
                        distance_method=atlas.distance_method,
                        bilateral=atlas.bilateral,
                        gpu_float32=gpu_float32,
                        device=dev)
                    return gpu.linalg.atlas_label_embedding_blocks(
                        blocks, atlas, end - start,
                        gpu_float32=gpu_float32, device=dev,
                        prepared=prepared_arg)

            with ThreadPoolExecutor(max_workers=len(shards)) as executor:
                shard_results = list(executor.map(_label_shard, shards))

            prefix_len = len(atlas.row_sum_1)
            embed = numpy.concatenate([r[0] for r in shard_results], axis=0)
            cluster_idx = numpy.concatenate([r[1] for r in shard_results], axis=0)
            dist = numpy.concatenate([r[2] for r in shard_results], axis=0)
            row_sum = numpy.concatenate(
                [atlas.row_sum_1] + [r[3][prefix_len:] for r in shard_results], axis=0)
            b_col_sums = numpy.concatenate([r[4] for r in shard_results], axis=0)
        else:
            # Stream similarity and embedding blocks; retain only CPU results.
            print(f"<{os.path.basename(__file__)}> Computing rectangular similarity matrix on GPU (fused embedding).")
            entry = _lookup_prepared_gpu(prepared_gpu, base_dev)
            if entry is not None:
                fixed_arg, prepared_arg = entry
            else:
                fixed_arg = _atlas_nystrom_coords(atlas)
                prepared_arg = None
            blocks = gpu.distance._iter_rectangular_similarity_arrays(
                queries, fixed_arg,
                atlas.threshold, atlas.sigma, distance_method=atlas.distance_method,
                bilateral=atlas.bilateral, gpu_float32=gpu_float32,
                device=base_dev)
            embed, cluster_idx, dist, row_sum, b_col_sums = \
                gpu.linalg.atlas_label_embedding_blocks(
                    blocks, atlas, number_fibers, gpu_float32=gpu_float32,
                    device=base_dev, prepared=prepared_arg)

    elif gpu.config.cpu_fast_enabled() and gpu.fiber_tensors.distance_supported(
            atlas.distance_method):

        print(f"<{os.path.basename(__file__)}> Computing streamed CPU similarity and embedding.")
        embed, cluster_idx, dist, row_sum, b_col_sums = gpu.cpu_backend.atlas_label(
            input_polydata, atlas, number_of_jobs=number_of_jobs, queries_15=queries_15)

    else:
        # 1) Compute fiber similarities (original joblib backend).
        nystrom_pd = _ensure_nystrom_polydata(atlas)
        B = \
            _rectangular_similarity_matrix(input_polydata, nystrom_pd,
                                           atlas.threshold, atlas.sigma, number_of_jobs, distance_method=atlas.distance_method,
                                           bilateral=atlas.bilateral, use_gpu=False,
                                           gpu_float32=gpu_float32)

        # 2) Do Normalized Cuts transform of similarity matrix.
        # row sum estimate for current B part of the matrix
        row_sum_2 = numpy.sum(B, axis=0) + \
            numpy.dot(atlas.row_sum_matrix, B)

        # in case of negative row sum estimation (this should not ever happen)
        if any(row_sum_2<=0):
            print(f"<{os.path.basename(__file__)}> Warning: Consider increasing sigma or using the Mean distance. negative row sum approximations.")
            print("Number of negative row sums:", numpy.count_nonzero(row_sum_2<=0))
            row_sum_2[row_sum_2<0] = 0.1

        # normalized cuts normalization
        row_sum = numpy.concatenate((atlas.row_sum_1, row_sum_2))
        dhat = numpy.sqrt(numpy.divide(1, row_sum))
        #dhat = numpy.sqrt(numpy.divide(1, numpy.concatenate((atlas.row_sum_1, row_sum_2))))
        B = \
            numpy.multiply(B, numpy.outer(dhat[0:sz], dhat[sz:].T))

        # 3) Compute eigenvectors for use in spectral embedding
        # <done already in atlas creation>

        # 4) Compute embedding using eigenvectors (slice top columns only)
        count = min(atlas.number_of_eigenvectors + 1, atlas.e_vec.shape[1])
        V = numpy.dot(
            numpy.dot(B.T, atlas.e_vec[:, -count:]),
            numpy.diag(numpy.divide(1.0, atlas.e_val[-count:])))
        V = numpy.divide(V, atlas.e_vec_norm[-count:])
        embed = numpy.zeros((number_fibers, atlas.number_of_eigenvectors))
        for i in range(0, atlas.number_of_eigenvectors):
            embed[:,i] = numpy.divide(V[:,-(i+2)], V[:,-1])

        # 5) Find clusters using k-means in embedding space.
        # Actually: LABEL using centroids from atlas
        print(f'<{os.path.basename(__file__)}> Cluster labeling in embedding space.')
        cluster_idx, dist = scipy.cluster.vq.vq(embed, atlas.centroids)
        b_col_sums = numpy.sum(B.T, axis=1)

    # 6) Output results.
    print(f'<{os.path.basename(__file__)}> Done labeling clusters, returning results.')
    # visualize embedding coordinates as RGB
    color = _embed_to_rgb(embed)
    # set up polydata with clustering output info.
    # for now modify input polydata by adding two arrays
    output_polydata = input_polydata
    output_polydata = \
        _format_output_polydata(output_polydata, cluster_idx, color, embed, row_sum, b_col_sums)

    return output_polydata, cluster_idx, color, embed

def _rectangular_distance_matrix(input_polydata_n, input_polydata_m, threshold,
                              number_of_jobs=3, landmarks_n=None, landmarks_m=None,
                              distance_method='Hausdorff', bilateral=False,
                              use_gpu=None, gpu_float32=None):

    """ Internal convenience function available to clustering
    routines.

    Computes distance matrix (nxm) for all n+m fibers in input
    polydata. each fiber in input_polydata_n is compared to each fiber
    in input_polydata_m.


    """

    if distance_method == 'Frechet':

        distances = similarity.rectangular_frechet_distances(input_polydata_n, input_polydata_m)
        distances = numpy.array(distances)

    elif gpu.use_gpu(use_gpu) and gpu.distance.distance_supported(
            distance_method, landmarks_n=landmarks_n, landmarks_m=landmarks_m):

        print(f"<{os.path.basename(__file__)}> Computing rectangular distance matrix on GPU.")
        # the GPU kernel returns (n, m); transpose to the same (m, n)
        # layout as the CPU path below
        distances = gpu.distance.rectangular_distance_matrix(
            input_polydata_n, input_polydata_m, threshold,
            distance_method=distance_method, bilateral=bilateral,
            gpu_float32=gpu_float32).T

    elif gpu.config.cpu_fast_enabled() and gpu.fiber_tensors.distance_supported(
            distance_method, landmarks_n=landmarks_n, landmarks_m=landmarks_m):

        # vectorized multi-threaded numpy backend: same numbers as the
        # joblib path below, much faster
        print(f"<{os.path.basename(__file__)}> Computing rectangular distance matrix with the vectorized CPU backend.")
        distances = gpu.cpu_backend.rectangular_distance_matrix(
            input_polydata_n, input_polydata_m, threshold,
            distance_method=distance_method, bilateral=bilateral).T

    else:
        
        fiber_array_n = fibers.FiberArray()
        fiber_array_n.convert_from_polydata(input_polydata_n, points_per_fiber=15)
        fiber_array_m = fibers.FiberArray()
        fiber_array_m.convert_from_polydata(input_polydata_m, points_per_fiber=15)

        if landmarks_n is None:
            landmarks_n = numpy.zeros((fiber_array_n.number_of_fibers,3))
    
        # pairwise distance matrix
        all_fibers_n = list(range(0, fiber_array_n.number_of_fibers))

        distances = Parallel(n_jobs=number_of_jobs,
                             verbose=0)(
            delayed(similarity.fiber_distance)(
                fiber_array_n.get_fiber(lidx),
                fiber_array_m,
                threshold, distance_method=distance_method,
                fiber_landmarks=landmarks_n[lidx,:], 
                landmarks=landmarks_m, bilateral=bilateral)
            for lidx in all_fibers_n)

        distances = numpy.array(distances).T

    return distances

def _rectangular_similarity_matrix(input_polydata_n, input_polydata_m, threshold, sigma,
                                number_of_jobs=3, landmarks_n=None, landmarks_m=None, distance_method='Hausdorff',
                                bilateral=False, use_gpu=None, gpu_float32=None):

    """ Internal convenience function available to clustering
    routines.

    Computes similarity matrix (nxn) for all n fibers in input
    polydata.  Calls function _pairwise_distance_matrix first.

    For distance computation and conversion to similarity uses
    parameters threshold and sigma.

    """

    if gpu.use_gpu(use_gpu) and gpu.distance.distance_supported(
            distance_method, landmarks_n=landmarks_n, landmarks_m=landmarks_m):

        print(f"<{os.path.basename(__file__)}> Computing rectangular similarity matrix on GPU.")
        # transpose to the same (m, n) layout as the CPU path below
        similarity_matrix = gpu.distance.rectangular_similarity_matrix(
            input_polydata_n, input_polydata_m, threshold, sigma,
            distance_method=distance_method, bilateral=bilateral,
            gpu_float32=gpu_float32).T
        return similarity_matrix

    distances = _rectangular_distance_matrix(input_polydata_n, input_polydata_m, threshold,
                                             number_of_jobs, landmarks_n, landmarks_m, distance_method,
                                             bilateral=bilateral, use_gpu=use_gpu, gpu_float32=gpu_float32)

    if distance_method == 'StrictSimilarity':
        similarity_matrix = distances
    else:
        # similarity matrix
        sigmasq = sigma * sigma
        similarity_matrix = similarity.distance_to_similarity(distances, sigmasq)

    return similarity_matrix

def _pairwise_distance_matrix(input_polydata, threshold,
                              number_of_jobs=3, landmarks=None, distance_method='Hausdorff',
                              bilateral=False, sigmasq=6400,
                              use_gpu=None, gpu_float32=None):

    """ Internal convenience function available to clustering
    routines.

    Computes distance matrix (nxn) for all n fibers in input
    polydata.

    """

    if distance_method == 'Frechet':

        distances = similarity.pairwise_frechet_distances(input_polydata,input_polydata)
        distances = numpy.array(distances)

    elif gpu.use_gpu(use_gpu) and gpu.distance.distance_supported(
            distance_method, landmarks=landmarks):

        print(f"<{os.path.basename(__file__)}> Computing pairwise distance matrix on GPU.")
        distances = gpu.distance.pairwise_distance_matrix(
            input_polydata, threshold, distance_method=distance_method,
            bilateral=bilateral, sigmasq=sigmasq, gpu_float32=gpu_float32)

    elif gpu.config.cpu_fast_enabled() and gpu.fiber_tensors.distance_supported(
            distance_method, landmarks=landmarks):

        print(f"<{os.path.basename(__file__)}> Computing pairwise distance matrix with the vectorized CPU backend.")
        distances = gpu.cpu_backend.pairwise_distance_matrix(
            input_polydata, threshold, distance_method=distance_method,
            bilateral=bilateral, sigmasq=sigmasq)

    else:

        fiber_array = fibers.FiberArray()
        fiber_array.convert_from_polydata(input_polydata, points_per_fiber=15)

        # pairwise distance matrix
        all_fibers = list(range(0, fiber_array.number_of_fibers))

        if landmarks is None:
            landmarks2 = numpy.zeros((fiber_array.number_of_fibers,3))
        else:
            landmarks2 = landmarks

        distances = Parallel(n_jobs=number_of_jobs,
                             verbose=0)(
            delayed(similarity.fiber_distance)(
                fiber_array.get_fiber(lidx),
                fiber_array,
                threshold, distance_method=distance_method, 
                fiber_landmarks=landmarks2[lidx,:], 
                landmarks=landmarks, bilateral=bilateral, sigmasq=sigmasq)
            for lidx in all_fibers)

        distances = numpy.array(distances)

        # remove outliers if desired????

    return distances

def _pairwise_similarity_matrix(input_polydata, threshold, sigma,
                                number_of_jobs=3, landmarks=None, distance_method='Hausdorff',
                                bilateral=False, sigmasq=6400,
                                use_gpu=None, gpu_float32=None):

    """ Internal convenience function available to clustering
    routines.

    Computes similarity matrix (nxn) for all n fibers in input
    polydata.  Calls function _pairwise_distance_matrix first.

    For distance computation and conversion to similarity uses
    parameters threshold and sigma.

    """

    if gpu.use_gpu(use_gpu) and gpu.distance.distance_supported(
            distance_method, landmarks=landmarks):

        print(f"<{os.path.basename(__file__)}> Computing pairwise similarity matrix on GPU.")
        similarity_matrix = gpu.distance.pairwise_similarity_matrix(
            input_polydata, threshold, sigma, distance_method=distance_method,
            bilateral=bilateral, sigmasq=sigmasq, gpu_float32=gpu_float32)

        # same sanity check as the CPU path: on-diagonal elements are all 1
        if __debug__:
            test = numpy.min(numpy.diag(similarity_matrix)) == 1.0
            if not test:
                print(f"<{os.path.basename(__file__)}> ERROR: On-diagonal elements are not all 1.0.")
                print(" Minimum on-diagonal value:", numpy.min(numpy.diag(similarity_matrix)))
                print(" Maximum on-diagonal value:", numpy.max(numpy.diag(similarity_matrix)))
                print(" Mean value:", numpy.mean(numpy.diag(similarity_matrix)))
                raise AssertionError
        return similarity_matrix

    distances = _pairwise_distance_matrix(input_polydata, threshold,
                                          number_of_jobs, landmarks, distance_method, bilateral=bilateral,
                                          use_gpu=use_gpu, gpu_float32=gpu_float32)
    
    if distance_method == 'StrictSimilarity':
        similarity_matrix = distances
    else:
        # similarity matrix
        sigmasq = sigma * sigma
        similarity_matrix = similarity.distance_to_similarity(distances, sigmasq)

    # sanity check that on-diagonal elements are all 1
    #print "This should be 1.0: ", numpy.min(numpy.diag(similarity_matrix))
    #print  numpy.min(numpy.diag(similarity_matrix)) == 1.0
    # test
    if __debug__:
        # this tests that on-diagonal elements are all 1
        test = numpy.min(numpy.diag(similarity_matrix)) == 1.0
        if not test:
            print(f"<{os.path.basename(__file__)}> ERROR: On-diagonal elements are not all 1.0.")
            print(" Minimum on-diagonal value:", numpy.min(numpy.diag(similarity_matrix)))
            print(" Maximum on-diagonal value:", numpy.max(numpy.diag(similarity_matrix)))
            print(" Mean value:", numpy.mean(numpy.diag(similarity_matrix)))
            raise AssertionError

    return similarity_matrix


def _format_output_polydata(output_polydata, cluster_idx, color, embed, estimated_row_sum, computed_column_sum):
    """ Output polydata with embedding colors, cluster numbers, and
    embedding coordinates.

    Cell data array names are:
    EmbeddingColor
    ClusterNumber
    EmbeddingCoordinate

    """
    # vectorized array construction (same value casts as the
    # InsertNext* calls of the original per-line loop)
    from vtk.util import numpy_support

    n_lines = output_polydata.GetNumberOfLines()

    embed_colors_arr = numpy.ascontiguousarray(
        numpy.asarray(color)[:n_lines, :3], dtype=numpy.uint8)
    embed_colors = numpy_support.numpy_to_vtk(embed_colors_arr, deep=True)
    embed_colors.SetName('EmbeddingColor')

    cluster_arr = numpy.ascontiguousarray(
        numpy.asarray(cluster_idx)[:n_lines], dtype=numpy.int32)
    cluster_colors = numpy_support.numpy_to_vtk(cluster_arr, deep=True)
    cluster_colors.SetName('ClusterNumber')

    embed_arr = numpy.ascontiguousarray(
        numpy.asarray(embed)[:n_lines, :], dtype=numpy.float32)
    embed_data = numpy_support.numpy_to_vtk(embed_arr, deep=True)
    embed_data.SetNumberOfComponents(embed_arr.shape[1])
    embed_data.SetName('EmbeddingCoordinate')

    row_sum_arr = numpy.ascontiguousarray(
        numpy.asarray(estimated_row_sum, dtype=numpy.float64).ravel()[:n_lines],
        dtype=numpy.float32)
    row_sum_data = numpy_support.numpy_to_vtk(row_sum_arr, deep=True)
    row_sum_data.SetName('TotalFiberSimilarity')

    col_arr = numpy.ascontiguousarray(
        numpy.asarray(computed_column_sum, dtype=numpy.float64).ravel()[:n_lines],
        dtype=numpy.float32)
    col_sum_data = numpy_support.numpy_to_vtk(col_arr, deep=True)
    col_sum_data.SetName('MeasuredFiberSimilarity')

    output_polydata.GetCellData().AddArray(embed_data)
    output_polydata.GetCellData().AddArray(cluster_colors)
    output_polydata.GetCellData().AddArray(embed_colors)
    output_polydata.GetCellData().AddArray(row_sum_data)
    output_polydata.GetCellData().AddArray(col_sum_data)

    output_polydata.GetCellData().SetActiveScalars('EmbeddingColor')

    return output_polydata


def _embed_to_rgb(embed):
    """

    Get colors from embedding (first 3 coordinates corresponding to
    max eigenvalues). Output minimum color is 0,0,0, max is
    255,255,255, (actually 180 due to brightness constraint) and all
    output values (brightnesses) are equal.

     """

    # make sure nothing is NaN
    testval = numpy.count_nonzero(numpy.isnan(embed))
    if testval:
        print(f"<{os.path.basename(__file__)}> Warning: Consider increasing sigma or using the Mean distance. NaN values encountered in embedding.")
        print("Number of NaN values:", testval)
        embed[numpy.isnan(embed)] = 0.0

    # first 3 components of embedding
    color = embed[:, 0:3]

    # normalize all colors to length 1
    color_len = numpy.sqrt(numpy.sum(numpy.power(color, 2), 1))
    #print numpy.min(color_len)
    color_len += 0.0001
    color = numpy.divide(color.T, color_len).T

    # convert to RGB: use colors from 0 to 255 for unsigned char (no
    # LUT in vtk). Before color components ranged from -1 to +1, now
    # from 0 to 255
    color = 127.5 + (color * 127.5)

    # Avoid dark black fibers that the cluster color is hard to see.
    # Convert to hsv, and make all brightness values the same.  That
    # way shadows show geometry, not cluster number.
    # Vectorized colorsys arithmetic, preserving branch priority and rounding.
    r, g, b = color.T
    maxc = numpy.maximum(numpy.maximum(r, g), b)
    minc = numpy.minimum(numpy.minimum(r, g), b)
    chromatic = minc != maxc
    delta = numpy.where(chromatic, maxc - minc, 1.)
    saturation = numpy.where(chromatic, (maxc - minc) / maxc, 0.)
    rc, gc, bc = (maxc - r) / delta, (maxc - g) / delta, (maxc - b) / delta
    hue = numpy.where(r == maxc, bc - gc,
                      numpy.where(g == maxc, 2. + rc - bc, 4. + gc - rc))
    hue = numpy.where(chromatic, (hue / 6.) % 1., 0.)
    sector = (hue * 6.).astype(numpy.int64)
    fraction = hue * 6. - sector
    p = 180. * (1. - saturation)
    q = 180. * (1. - saturation * fraction)
    t = 180. * (1. - saturation * (1. - fraction))
    sector %= 6
    cases = [sector == i for i in range(6)]
    color[:, 0] = numpy.select(cases, [180., q, p, p, t, 180.])
    color[:, 1] = numpy.select(cases, [t, 180., 180., q, p, p])
    color[:, 2] = numpy.select(cases, [p, p, t, 180., 180., q])

    # To look at this transform of the embed data (sanity check)
    #plt.figure()
    #plt.plot(embed[:, 0], color[:, 0], 'r.')
    #plt.plot(embed[:, 1], color[:, 1], 'g.')
    #plt.plot(embed[:, 2], color[:, 2], 'b.')

    return color


def compute_cluster_loo_similarity(pd_c, subject_ids, sigma, distance_method='Mean',
                                   bilateral=False, use_gpu=None, gpu_float32=None,
                                   return_distances=False):
    """Compute leave-one-subject-out mean similarity for every fiber in one cluster."""
    subject_arr = numpy.asarray(subject_ids)
    n_fibers = int(subject_arr.shape[0])
    sigmasq = float(sigma) * float(sigma)
    if n_fibers == 0:
        empty = numpy.zeros(0, dtype=numpy.float64)
        if return_distances:
            empty_m = numpy.zeros((0, 0), dtype=numpy.float64)
            return empty, empty_m, empty_m
        return empty

    if (not return_distances and distance_method == 'Mean'
            and gpu.use_gpu(use_gpu)
            and gpu.distance.distance_supported(distance_method)):
        X = gpu.fiber_tensors.polydata_to_numpy(pd_c, 15)
        total_sums = gpu.distance.similarity_row_sums_from_arrays(
            X, X, 0.0, distance_method=distance_method, bilateral=bilateral,
            sigmasq=sigmasq, gpu_float32=gpu_float32)
        total_similarity = numpy.empty(n_fibers, dtype=numpy.float64)
        uniq_subjects, inverse = numpy.unique(subject_arr, return_inverse=True)
        for u_idx in range(len(uniq_subjects)):
            s_idx = numpy.flatnonzero(inverse == u_idx)
            n_s = int(s_idx.size)
            n_loo = n_fibers - n_s
            if n_loo == 0:
                total_similarity[s_idx] = numpy.nan
            elif n_s == 1:
                total_similarity[s_idx] = (total_sums[s_idx] - 1.0) / float(n_loo)
            else:
                X_s = X[s_idx]
                same_sums = gpu.distance.similarity_row_sums_from_arrays(
                    X_s, X_s, 0.0, distance_method=distance_method, bilateral=bilateral,
                    sigmasq=sigmasq, gpu_float32=gpu_float32)
                total_similarity[s_idx] = (total_sums[s_idx] - same_sums) / float(n_loo)
        return total_similarity

    cluster_distances = _pairwise_distance_matrix(
        pd_c, 0.0, number_of_jobs=1, bilateral=bilateral,
        distance_method=distance_method, sigmasq=sigmasq,
        use_gpu=use_gpu, gpu_float32=gpu_float32)
    if distance_method == 'StrictSimilarity':
        cluster_similarity = cluster_distances
    else:
        cluster_similarity = similarity.distance_to_similarity(cluster_distances, sigmasq)

    row_sums = numpy.sum(cluster_similarity, axis=1)
    total_similarity = numpy.empty(n_fibers, dtype=numpy.float64)
    uniq_subjects, inverse = numpy.unique(subject_arr, return_inverse=True)
    for u_idx in range(len(uniq_subjects)):
        s_idx = numpy.flatnonzero(inverse == u_idx)
        n_loo = n_fibers - int(s_idx.size)
        if n_loo == 0:
            total_similarity[s_idx] = numpy.nan
        else:
            same_sums = numpy.sum(cluster_similarity[numpy.ix_(s_idx, s_idx)], axis=1)
            total_similarity[s_idx] = (row_sums[s_idx] - same_sums) / float(n_loo)

    if return_distances:
        return total_similarity, cluster_distances, cluster_similarity
    return total_similarity


def output_and_quality_control_cluster_atlas(atlas, output_polydata_s, subject_fiber_list, input_polydatas, number_of_subjects, outdir, cluster_numbers_s, color, embed, number_of_fibers_to_display, testing=False, verbose=False, render_images=True, number_of_jobs=1):

    """Save the output in our atlas format for automatic labeling of clusters.

    First save the atlas.vtp and atlas.p datasets. This is the data used to
    label a new subject.  Also write the polydata with cluster indices
    saved as cell data. This is a one-file output option for clusters.
    Finally, save some quality control metrics and save the atlas
    clusters as individual polydatas. This is used to set up a mrml
    hierarchy file and to visualize the output in Slicer. This data is
    not used to label a new subject.
    """

    # Write additional output for software testing if code has changed (requires random seed to be constant)
    if testing:
        expected_results_file = os.path.join(outdir, 'test_cluster_atlas_numbers.pkl')
        pickle.dump(cluster_numbers_s, open(expected_results_file, 'wb'))
        expected_results_file = os.path.join(outdir, 'test_cluster_atlas_colors.pkl')
        pickle.dump(color, open(expected_results_file, 'wb'))
        expected_results_file = os.path.join(outdir, 'test_cluster_atlas_embeddings.pkl')
        pickle.dump(embed, open(expected_results_file, 'wb'))
         
    # Save the output in our atlas format for automatic labeling of full brain datasets.
    # This is the data used to label a new subject
    atlas.save(outdir,'atlas')

    # Write the polydata with cluster indices saved as cell data
    fname_output = os.path.join(outdir, 'clustered_whole_brain.vtp')
    io.write_polydata(output_polydata_s, fname_output)

    # output summary file to save information about all subjects
    subjects_qc_fname = os.path.join(outdir, 'input_subjects.txt')
    subjects_qc_file = open(subjects_qc_fname, 'w')
    outstr = "Subject_idx\tSubject_ID\tfilename\n"
    subjects_qc_file.write(outstr)
    idx = 1
    for fname in input_polydatas:
        subject_id = os.path.splitext(os.path.basename(fname))[0]
        outstr =  str(idx) + '\t' + str(subject_id) + '\t' + str(fname) + '\n'
        subjects_qc_file.write(outstr)
        idx += 1
    subjects_qc_file.close()

    # output summary file to save information about all clusters
    clusters_qc_fname = os.path.join(outdir, 'cluster_quality_control.txt')

    # Figure out how many subjects in each cluster (ideally, most subjects in most clusters)
    subjects_per_cluster = list()
    percent_subjects_per_cluster = list()
    mean_fiber_len_per_cluster = list()
    std_fiber_len_per_cluster = list()
    mean_fibers_per_subject_per_cluster = list()
    std_fibers_per_subject_per_cluster = list()

    # find out length of each fiber
    fiber_length, step_size = filter.compute_lengths(output_polydata_s)

    # loop over each cluster and compute quality control metrics
    cluster_indices = list(range(atlas.centroids.shape[0]))
    subj_arr = numpy.asarray(subject_fiber_list, dtype=numpy.int64)
    for cidx in cluster_indices:
        cluster_mask = (cluster_numbers_s == cidx)
        subj_c = subj_arr[cluster_mask]
        subjects_per_cluster.append(len(set(subj_c)))
        fibers_per_subject = numpy.bincount(subj_c, minlength=number_of_subjects)
        mean_fibers_per_subject_per_cluster.append(numpy.mean(fibers_per_subject))
        std_fibers_per_subject_per_cluster.append(numpy.std(fibers_per_subject))
        mean_fiber_len_per_cluster.append(numpy.mean(fiber_length[cluster_mask]))
        std_fiber_len_per_cluster.append(numpy.std(fiber_length[cluster_mask]))

    percent_subjects_per_cluster = numpy.divide(numpy.array(subjects_per_cluster), float(number_of_subjects))

    # Save output quality control information
    print(f"<{os.path.basename(__file__)}> Saving cluster quality control information file.")
    with open(clusters_qc_fname, 'w') as clusters_qc_file:
        print('cluster_idx', '\t', 'number_subjects', '\t', 'percent_subjects', '\t', 'mean_length', '\t', 'std_length', '\t', 'mean_fibers_per_subject', '\t', 'std_fibers_per_subject', file=clusters_qc_file)
        for cidx in cluster_indices:
            print(cidx + 1, '\t', subjects_per_cluster[cidx], '\t', percent_subjects_per_cluster[cidx] * 100.0, '\t',
                  mean_fiber_len_per_cluster[cidx], '\t', std_fiber_len_per_cluster[cidx], '\t',
                  mean_fibers_per_subject_per_cluster[cidx], '\t', std_fibers_per_subject_per_cluster[cidx], file=clusters_qc_file)

    if HAVE_PLT:
        print(f"<{os.path.basename(__file__)}> Saving subjects per cluster histogram.")
        fig, ax = plt.subplots()
        counts = numpy.zeros(number_of_subjects+1)
        counts[:numpy.max(subjects_per_cluster)+1] = numpy.bincount(subjects_per_cluster)
        ax.bar(list(range(number_of_subjects + 1)), counts, width=1, align='center')
        ax.set(xlim=[-1, number_of_subjects + 1])
        plt.title('Histogram of Subjects per Cluster')
        plt.xlabel('subjects per cluster')
        plt.ylabel('number of clusters')
        plt.savefig( os.path.join(outdir, 'subjects_per_cluster_hist.pdf'))
        plt.close()
        
    # Save the entire combined atlas as individual clusters for visualization
    # and labeling/naming of structures. This will include all of the data
    # that was clustered to make the atlas.

    # Figure out file name and mean color for each cluster, and write the individual polydatas
    pd_c_list = mask_all_clusters(output_polydata_s, cluster_numbers_s, len(cluster_indices), preserve_point_data=True,
                                  preserve_cell_data=True, verbose=False, number_of_jobs=number_of_jobs)
    print(f"<{os.path.basename(__file__)}> Beginning to save individual clusters as polydata files. TOTAL CLUSTERS:", len(cluster_indices), end=' ')
    fnames = list()
    cluster_colors = list()
    cluster_sizes = list()
    cluster_fnames = list()
    for c in cluster_indices:
        mask = cluster_numbers_s == c
        cluster_size = numpy.sum(mask)
        cluster_sizes.append(cluster_size)
        pd_c = pd_c_list[c]
        filter.add_point_data_array(pd_c, subject_fiber_list[mask], "Subject_ID")
        farray = fibers.FiberArray()
        farray.hemispheres = True
        farray.hemisphere_percent_threshold = 0.90
        farray.convert_from_polydata(pd_c, points_per_fiber=50)
        filter.add_point_data_array(pd_c, farray.fiber_hemisphere, "Hemisphere")
        fname_c = f'cluster_{c+1:05d}.vtp'
        fnames.append(fname_c)
        cluster_fnames.append(os.path.join(outdir, fname_c))
        if cluster_size > 0:
            color_c = color[mask, :]
            cluster_colors.append(numpy.mean(color_c, 0))
        else:
            cluster_colors.append([0, 0, 0])
    io.write_polydatas_parallel(
        pd_c_list, cluster_fnames, jobs=number_of_jobs, use_threads=bool(gpu.use_gpu()))
    del pd_c_list
    print("\n<cluster.py> Finishes saving individual clusters as polydata files.")

    # Notify user if some clusters empty
    empty_count = 0
    for sz, fname in zip(cluster_sizes,cluster_fnames):
        if sz == 0:
            print(sz, ":", fname)
            empty_count += 1
    if empty_count:
        print(f"<{os.path.basename(__file__)}> Warning. Empty clusters found:", empty_count)

    cluster_sizes = numpy.array(cluster_sizes)
    print(f"<{os.path.basename(__file__)}> Mean number of fibers per cluster:", numpy.mean(cluster_sizes), "Range:", numpy.min(cluster_sizes), "..", numpy.max(cluster_sizes))

    # Estimate subsampling ratio to display approximately number_of_fibers_to_display total fibers in 3D Slicer
    number_fibers = len(cluster_numbers_s)
    if number_fibers < number_of_fibers_to_display:
        ratio = 1.0
    else:
        ratio = number_of_fibers_to_display / number_fibers
    print(f"<{os.path.basename(__file__)}> Subsampling ratio for display of", number_of_fibers_to_display, "total fibers estimated as:", ratio)

    # Write the MRML file into the directory where the polydatas were already stored
    fname = os.path.join(outdir, 'clustered_tracts.mrml')
    mrml.write(fnames, numpy.around(numpy.array(cluster_colors), decimals=3), fname, ratio=ratio)

    # Also write one with 100% of fibers displayed
    fname = os.path.join(outdir, 'clustered_tracts_display_100_percent.mrml')
    mrml.write(fnames, numpy.around(numpy.array(cluster_colors), decimals=3), fname, ratio=1.0)
    
    # View the whole thing in jpg format for quality control
    if render_images:
        print(f'<{os.path.basename(__file__)}> Rendering and saving images of cluster atlas.')
        ren = render.render(output_polydata_s, 1000, data_mode='Cell', data_name='EmbeddingColor', verbose=verbose)
        ren.save_views(outdir, verbose=verbose)
        del ren

def mask_all_clusters(inpd, cluster_numbers_s, number_of_clusters, color=None,
                      preserve_point_data=True, preserve_cell_data=False, verbose=True,
                      number_of_jobs=1):
    """Split fibers in original order using the vectorized line selector."""
    print(f'<{os.path.basename(__file__)}> Masking all clusters: total ', number_of_clusters)
    labels = numpy.asarray(cluster_numbers_s)
    valid = labels >= 0
    indices = labels.astype(numpy.int64)
    if numpy.any(indices[valid] >= number_of_clusters):
        raise IndexError("Cluster index exceeds number_of_clusters")
    geometry = filter._lines_offsets_connectivity(inpd)
    if geometry is not None:
        order = numpy.flatnonzero(valid)
        order = order[numpy.argsort(indices[order], kind='stable')]
        boundaries = numpy.searchsorted(indices[order], numpy.arange(number_of_clusters + 1))
        workers = int(number_of_jobs) if number_of_jobs is not None else 1
        if workers > 1 and number_of_clusters > 1 and not verbose:
            from concurrent.futures import ThreadPoolExecutor

            point_specs, cell_specs, tensor_names = filter._extract_polydata_schema_and_arrays(
                inpd, preserve_point_data=preserve_point_data, preserve_cell_data=preserve_cell_data)

            def _build_one(c):
                return filter._build_polydata_from_indexed_lines(
                    geometry, order[boundaries[c]:boundaries[c + 1]],
                    point_specs=point_specs, cell_specs=cell_specs, tensor_names=tensor_names)

            with ThreadPoolExecutor(max_workers=min(workers, number_of_clusters)) as pool:
                return list(pool.map(_build_one, range(number_of_clusters)))
        return [filter.mask(inpd, None, preserve_point_data=preserve_point_data,
                            preserve_cell_data=preserve_cell_data, verbose=verbose,
                            _geometry=geometry, _line_indices=order[boundaries[c]:boundaries[c + 1]])
                for c in range(number_of_clusters)]
    # The original splitter ignores color and disables active cell scalars.
    return [filter.mask(inpd, valid & (indices == c),
                        preserve_point_data=preserve_point_data,
                        preserve_cell_data=preserve_cell_data, verbose=verbose)
            for c in range(number_of_clusters)]
