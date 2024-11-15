'''k_means.py
Performs K-Means clustering
Ruby Nunez
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors


class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None

        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None

        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data

        # num_samps: int. Number of samples in the dataset
        self.num_samps = None

        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        
        if data is not None:
            self.num_samps, self.num_features = data.shape


    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        self.data = data


    def get_data(self):
        '''Gets a copy of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features).
        '''

        return self.data.copy()


    def get_centroids(self):
        '''Get the K-means centroids

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''

        return self.centroids


    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''

        return self.data_centroid_labels


    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Computes the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.
        '''

        euclidean_dist = np.sqrt(np.sum(np.square(pt_1 - pt_2)))

        return euclidean_dist


    def dist_pt_to_centroids(self, pt, centroids):
        '''Computes the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.
        '''
        
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)

        dist_bt_centroids = np.array([self.dist_pt_to_pt(centroid, pt) for centroid in centroids])

        return dist_bt_centroids


    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.
        '''

        self.k = k

        random_indices = np.random.choice(self.num_samps, size=k, replace=False)

        return self.data[random_indices]


    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Makes sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Prints out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for
        '''

        self.initialize(k)

        num_iterations = 0
        centroid_diff = tol + 1

        centroid_diff = np.zeros(k)
        prev_centroids = self.centroids.copy()

        for num_iterations in range(max_iter):
            self.data_centroid_labels = self.update_labels(self.centroids)
            self.centroids, centroid_diff = self.update_centroids(k, self.data_centroid_labels, prev_centroids)

            if np.max(abs(centroid_diff)) <= tol:
                break

            prev_centroids = self.centroids.copy()

        self.inertia = self.compute_inertia()

        if verbose:
            print("Total number of iterations:", num_iterations)

        return self.inertia, num_iterations


    def cluster_batch(self, k=2, n_iter=5, verbose=False):
        '''Runs K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''

        best_inertia = float('inf')
        best_centroids = None
        best_labels = None

        for _ in range(n_iter):
            self.initialize(k)
            inertia, _ = self.cluster(k, verbose=verbose)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = self.centroids.copy()
                best_labels = self.data_centroid_labels.copy()

        self.centroids = best_centroids
        self.data_centroid_labels = best_labels
        self.inertia = best_inertia


    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample.
        '''
        labels = []

        for data_point in self.data:
            distances = [self.dist_pt_to_pt(data_point, centroid) for centroid in centroids]
            min_distance_index = distances.index(min(distances))
            labels.append(min_distance_index)
        
        nparr_labels = np.array(labels)

        return nparr_labels


    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values
        '''

        new_centroids = np.zeros_like(prev_centroids)

        centroid_diff = np.zeros_like(prev_centroids)

        for i in range(k):
            cluster_samples = self.data[data_centroid_labels == i]            

            if len(cluster_samples) > 0:
                new_centroids[i] = np.mean(cluster_samples, axis=0)
                centroid_diff[i] = new_centroids[i] - prev_centroids[i]
            else:
                random_sample_idx = np.random.choice(self.num_samps)
                new_centroids[i] = self.data[random_sample_idx]
                centroid_diff[i] = self.dist_pt_to_pt(prev_centroids[i], new_centroids[i])

        self.centroids = new_centroids

        return new_centroids, centroid_diff


    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''

        squared_distances = []

        for i in range(self.num_samps):
            pt = self.data[i]
            assigned_centroid = self.centroids[self.data_centroid_labels[i]]
            distance = self.dist_pt_to_centroids(pt, assigned_centroid)
            squared_distance = distance ** 2
            squared_distances.append(squared_distance)

        inertia = np.mean(squared_distances)

        return inertia


    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        k = len(np.unique(self.data_centroid_labels))

        palette = cartocolors.qualitative.Safe_10.mpl_colors

        for i in range(k):
            cluster_data = self.data[self.data_centroid_labels == i]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=[palette[i]], label=f'Cluster {i+1}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', s=100, c='black', label='Centroids')

        plt.title('K-means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(prop={'size': 8})


    def elbow_plot(self, max_k, n_iter=1):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        '''

        initial_centroids = self.initialize(max_k)
        new_labels = self.update_labels(initial_centroids)
        self.update_centroids(max_k, new_labels, initial_centroids)

        inertias = []
        for k in range(1, max_k + 1):
            self.cluster_batch(k=k, n_iter=n_iter)
            inertias.append(self.compute_inertia())

        plt.plot(range(1, max_k + 1), inertias, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.xticks(range(1, max_k + 1))
        plt.title('Elbow Plot')


    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''

        closest_centroid_indices = []

        for pixel in self.data:

            distances = self.dist_pt_to_centroids(pixel, self.centroids)

            closest_centroid_index = 0
            min_distance = float('inf')
            for i, distance in enumerate(distances):
                if distance < min_distance:
                    closest_centroid_index = i
                    min_distance = distance
            
            closest_centroid_indices.append(closest_centroid_index)

        self.data = np.array([self.centroids[index] for index in closest_centroid_indices])