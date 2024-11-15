'''rbf_net.py
Radial Basis Function Neural Network
Ruby Nunez
'''

import numpy as np
import k_means 
import scipy


class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset
        '''

        # k: Number of hidden units
        self.k = num_hidden_units
        
        # num_classes: Number of output units
        self.num_classes = num_classes

        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None

        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None

        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None


    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''

        return self.prototypes


    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''

        return self.k


    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''

        return self.num_classes


    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.
        '''

        kmeans_obj = k_means.KMeans(data)

        kmeans_obj.initialize(k=centroids.shape[0])

        self.prototypes = centroids

        self.k = kmeans_obj.k

        avg_distances = np.zeros(self.k)

        for i in range(self.k):
            cluster_indices = np.where(cluster_assignments == i)[0]
            cluster_data = data[cluster_indices]
            centroid = centroids[i]
            distances = np.zeros(len(cluster_data))
            for j in range(len(cluster_data)):
                distances[j] = kmeans_obj.dist_pt_to_centroids(cluster_data[j], centroid)
            avg_distances[i] = np.mean(distances)

        return avg_distances


    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        '''

        kmeans_obj = k_means.KMeans(data)
        kmeans_obj.initialize(self.k)
        kmeans_obj.centroids = self.prototypes
        kmeans_obj.cluster_batch(self.k, n_iter=5, verbose=False)
        self.cluster_assignments = kmeans_obj.get_data_centroid_labels()
        self.prototypes = kmeans_obj.centroids
        self.sigmas = self.avg_cluster_dist(data, self.prototypes, self.cluster_assignments, kmeans_obj)


    def linear_regression(self, A, y):
        '''Performs linear regression

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term
        '''

        Ahat = np.hstack((A, np.ones((A.shape[0], 1))))

        c, _, _, _ = np.linalg.lstsq(Ahat, y)

        return c


    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
        '''

        num_samps = data.shape[0]
        hidden_activation = np.zeros((num_samps, self.k))

        for i in range(num_samps):
            for j in range(self.k):
                distance = np.linalg.norm(data[i] - self.prototypes[j]) ** 2
                hidden_activation[i][j] = np.exp(-distance / (2 * self.sigmas[j] ** 2))

        return hidden_activation


    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.
        '''

        num_samps = hidden_acts.shape[0]
        bias_unit = np.ones((num_samps, 1))
        hidden_acts_with_bias = np.hstack((hidden_acts, bias_unit))

        output_activation = np.dot(hidden_acts_with_bias, self.wts)

        return output_activation


    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.
        '''

        self.initialize(data)
        recoded_y = np.zeros((y.shape[0], self.num_classes))
        for c in range(self.num_classes):
            recoded_y[:, c] = (y == c).astype(int)
        hidden_acts = self.hidden_act(data)
        self.wts = self.linear_regression(hidden_acts, recoded_y)


    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.
        '''

        hidden_acts = self.hidden_act(data)
        output_acts = self.output_act(hidden_acts)
        predicted_classes = np.argmax(output_acts, axis=1)

        return predicted_classes


    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.
        '''

        num_correct = np.sum(y == y_pred)
        accuracy = num_correct / len(y)

        return accuracy

