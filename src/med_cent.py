import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors

import warnings
import numpy as np
from scipy import sparse as sp

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.sparsefuncs import csc_median_axis_0
from sklearn.utils.multiclass import check_classification_targets

class median_centroid(NearestNeighbors):
    def __init__(self, metric="euclidean", *, shrink_threshold=None):
        #print("Hello")
        self.metric = metric
        self.shrink_threshold = shrink_threshold


    def _median(self,_X):
        return np.median(_X, axis=0)

    def fit(self, X, y):
        #print(X)
        """
        Fit the NearestCentroid model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
            Note that centroid shrinking cannot be used with sparse matrices.
        y : array-like of shape (n_samples,)
            Target values (integers)
        """
        if self.metric == "precomputed":
            raise ValueError("Precomputed is not supported.")
        # If X is sparse and the metric is "manhattan", store it in a csc
        # format is easier to calculate the median.
        if self.metric == "manhattan":
            X, y = self._validate_data(X, y, accept_sparse=["csc"])
        else:
            X, y = self._validate_data(X, y, accept_sparse=["csr", "csc"])
        is_X_sparse = sp.issparse(X)
        if is_X_sparse and self.shrink_threshold:
            raise ValueError("threshold shrinking not supported for sparse input")
        check_classification_targets(y)

        n_samples, n_features = X.shape
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self.classes_ = classes = le.classes_
        #print("Classes: ",le.classes_)
        n_classes = classes.size
        if n_classes < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % (n_classes)
            )

        # Mask mapping each class to its members.
        self.centroids_ = np.empty((n_classes, n_features), dtype=np.float64)
        # Number of clusters in each class.
        nk = np.zeros(n_classes)

        for cur_class in range(n_classes):
            center_mask = y_ind == cur_class
            nk[cur_class] = np.sum(center_mask)
            if is_X_sparse:
                center_mask = np.where(center_mask)[0]

            # XXX: Update other averaging methods according to the metrics.
            if self.metric == "manhattan":
                # NumPy does not calculate median of sparse matrices.
                if not is_X_sparse:
                    self.centroids_[cur_class] = np.median(X[center_mask], axis=0)
                else:
                    self.centroids_[cur_class] = csc_median_axis_0(X[center_mask])
            else:
                if self.metric != "euclidean":
                    warnings.warn(
                        "Averaging for metrics other than "
                        "euclidean and manhattan not supported. "
                        "The average is set to be the mean."
                    )
                #self.centroids_[cur_class] = X[center_mask].mean(axis=0)
                self.centroids_[cur_class] = self._median(X[center_mask]) #Here they are calculating class centroid.
                #print("printing from mc")
                #print(X[center_mask])
                #print(self.centroids_[cur_class])
                #print("Median: ",self._median(X[center_mask]))

        if self.shrink_threshold:
            if np.all(np.ptp(X, axis=0) == 0):
                #print("Yo BHASKAR")
                return self
                #raise ValueError("All features have zero variance. Division by zero.")
            dataset_centroid_ = np.mean(X, axis=0)
            #dataset_centroid_ = np.median(X, axis=0)

            # m parameter for determining deviation
            m = np.sqrt((1.0 / nk) - (1.0 / n_samples))
            # Calculate deviation using the standard deviation of centroids.
            variance = (X - self.centroids_[y_ind]) ** 2
            variance = variance.sum(axis=0)
            s = np.sqrt(variance / (n_samples - n_classes))
            s += np.median(s)  # To deter outliers from affecting the results.
            mm = m.reshape(len(m), 1)  # Reshape to allow broadcasting.
            ms = mm * s
            deviation = (self.centroids_ - dataset_centroid_) / ms
            # Soft thresholding: if the deviation crosses 0 during shrinking,
            # it becomes zero.
            signs = np.sign(deviation)
            deviation = np.abs(deviation) - self.shrink_threshold
            np.clip(deviation, 0, None, out=deviation)
            deviation *= signs
            # Now adjust the centroids using the deviation
            msd = ms * deviation
            self.centroids_ = dataset_centroid_[np.newaxis, :] + msd
        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.
        The predicted class C for each sample in X is returned.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        C : ndarray of shape (n_samples,)
        Notes
        -----
        If the metric constructor parameter is "precomputed", X is assumed to
        be the distance matrix between the data to be predicted and
        ``self.centroids_``.
        """
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        return self.classes_[
            pairwise_distances(X, self.centroids_, metric=self.metric).argmin(axis=1)
        ]

if __name__ == "__main__":
    X = np.array([[1, -1], 
                [2, -1],
                [2, -2],
                [4, 1],

                [5, 1],
                [4, 1],
                [5, 2],
                [5, 2]])

    y = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    clf = median_centroid()
    clf.fit(X, y)
    #print(clf.predict([[-0.8, -1]]))
    print(clf.centroids_)