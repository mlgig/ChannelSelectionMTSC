import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())
#from sklearn.feature_selection import SequentialFeatureSelector
import time

from sktime.transformations.panel.rocket import Rocket
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from  src.classelbow import ElbowPair

"""
Sequential feature selection
"""
import numbers

import numpy as np

from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifierCV
from src.dataset import dataset_asc
from sklearn.metrics import accuracy_score


import numbers

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils._tags import _safe_tags
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_val_score

class SequentialFeatureSelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator):

    def __init__(
        self,
        estimator,
        *,
        n_features_to_select=None,
        direction="forward",
        scoring=None,
        cv=5,
        n_jobs=None,
    ):

        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Learn the features to select from X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.
        y : array-like of shape (n_samples,), default=None
            Target values. This parameter may be ignored for
            unsupervised learning.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        tags = self._get_tags()
        """X = self._validate_data(
            X,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
        )"""
        n_features = X.shape[1]

        error_msg = (
            "n_features_to_select must be either None, an "
            "integer in [1, n_features - 1] "
            "representing the absolute "
            "number of features, or a float in (0, 1] "
            "representing a percentage of features to "
            f"select. Got {self.n_features_to_select}"
        )
        if self.n_features_to_select is None:
            self.n_features_to_select_ = n_features // 2
        elif isinstance(self.n_features_to_select, numbers.Integral):
            if not 0 < self.n_features_to_select < n_features:
                raise ValueError(error_msg)
            self.n_features_to_select_ = self.n_features_to_select
        elif isinstance(self.n_features_to_select, numbers.Real):
            if not 0 < self.n_features_to_select <= 1:
                raise ValueError(error_msg)
            self.n_features_to_select_ = int(n_features * self.n_features_to_select)
        else:
            raise ValueError(error_msg)

        if self.direction not in ("forward", "backward"):
            raise ValueError(
                "direction must be either 'forward' or 'backward'. "
                f"Got {self.direction}."
            )

        cloned_estimator = clone(self.estimator)

        # the current mask corresponds to the set of features:
        # - that we have already *selected* if we do forward selection
        # - that we have already *excluded* if we do backward selection
        current_mask = np.zeros(shape=n_features, dtype=bool)
        n_iterations = (
            self.n_features_to_select_
            if self.direction == "forward"
            else n_features - self.n_features_to_select_
        )
        for _ in range(n_iterations):
            new_feature_idx = self._get_best_new_feature(
                cloned_estimator, X, y, current_mask
            )
            current_mask[new_feature_idx] = True

        if self.direction == "backward":
            current_mask = ~current_mask
        self.support_ = current_mask

        return self

    def _get_best_new_feature(self, estimator, X, y, current_mask):
        # Return the best new feature to add to the current_mask, i.e. return
        # the best new feature to add (resp. remove) when doing forward
        # selection (resp. backward selection)
        candidate_feature_indices = np.flatnonzero(~current_mask)
        #print(current_mask)
        scores = {}
        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            if self.direction == "backward":
                candidate_mask = ~candidate_mask
            X_new = X.iloc[:, candidate_mask]
            scores[feature_idx] = cross_val_score(
                estimator,
                X_new,
                y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            ).mean()
        return max(scores, key=lambda feature_idx: scores[feature_idx])

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    def _more_tags(self):
        return {
            "allow_nan": _safe_tags(self.estimator, key="allow_nan"),
            "requires_y": True,
        }

if __name__ == '__main__':

    rocket = Rocket(random_state=0)

    model = Pipeline(
                [
                ('rocket', Rocket(random_state=0)),
                ('model', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),normalize=True, class_weight='balanced'))
                ])


    #toc_fwd = time.time()
    #dataset_ = ["PenDigits"]
    #print(dataset_asc[::-1])
    for item in dataset_asc[::-1]:
        #print(item)

        train_x, train_y = load_from_tsfile_to_dataframe(f"./data/{item}/{item}_TRAIN.ts", return_separate_X_and_y=True)
        test_x, test_y = load_from_tsfile_to_dataframe(f"./data/{item}/{item}_TEST.ts", return_separate_X_and_y=True)
        
        start = time.time()
                
        #print(f"{item} \nShape: {train_x.shape} ")
        
        obj = ElbowPair(distance='eu', shrinkage=0, center="mean")
        obj.fit(train_x, train_y)
        chs = len(obj.relevant_dims)
        #print(f"number channel to be selected: {chs}")
        if chs==train_x.shape[1]:
            chs=chs-1
        sfs_forward = SequentialFeatureSelector(model, n_features_to_select=chs, direction="forward", scoring='accuracy', n_jobs=-2, cv=3).fit(train_x, train_y)
        
        model.fit(train_x.iloc[:,sfs_forward.get_support()], train_y)
        end = time.time()
        
        preds = model.predict(test_x.iloc[:, sfs_forward.get_support()])
        acc1 = accuracy_score(preds, test_y) * 100

        df = pd.DataFrame({"Dataset": item, "Acc": [acc1], "time (min)":[(end-start)/60]})
        df.to_csv(f"./benchmark/{item}.csv", index=False)
        print(df)

    
    

