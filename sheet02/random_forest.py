import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        max_features='sqrt',
    ):
        """
        Parameters
        ----------
        n_estimators      : number of trees in the forest
        max_depth         : maximum depth of each tree
        min_samples_split : minimum samples required to split a node
        max_features      : features considered at each split
                            'sqrt' (default) → sqrt(n_features)
                            int              → fixed number of features
                            None             → all features (= bagged trees)
        """
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.trees_            = []

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X_train, y_train):
        self.trees_ = []
        n_samples   = X_train.shape[0]

        for i in range(self.n_estimators):
            # Bootstrap sample (sample with replacement)
            bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_train[bootstrap_idx]
            y_boot = y_train[bootstrap_idx]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)



    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        # Collect predictions from every tree: shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.trees_], dtype=int)

        # Majority vote
        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            predictions[i] = np.bincount(all_preds[:, i]).argmax()
        return predictions