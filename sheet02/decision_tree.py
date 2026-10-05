import numpy as np



class Node:
    def __init__(self):
        self.left_child  = None
        self.right_child = None
        self.is_leaf     = False
        self.feature     = None
        self.threshold   = None
        self.entropy     = None
        self.samples     = None
        self.labels      = None
        self.pred        = None
        self.depth       = None



class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=10, max_features=None):
        """
        Parameters
        ----------
        max_depth          : maximum depth of the tree
        min_samples_split  : minimum samples required to split a node
        max_features       : number of features to consider at each split.
                             None → use all features
                             int  → use that many randomly sampled features(Question2)
                             'sqrt' → use sqrt(n_features)(Question2)
        """
        self.root              = None
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features


    def entropy(self, labels):
        entropy = 0
        _, counts = np.unique(labels, return_counts=True)
        for count in counts:
            prob = count / len(labels)
            entropy += prob * np.log2(prob)
        return -entropy

    
    def information_gain(self, node, left_labels, right_labels):
        left_entropy  = self.entropy(left_labels)
        right_entropy = self.entropy(right_labels)

        total_samples = len(node.labels)
        weighted_child = (
            (len(left_labels)  / total_samples) * left_entropy +
            (len(right_labels) / total_samples) * right_entropy
        )
        return node.entropy - weighted_child


    def best_split(self, node):
        best_gain      = -1
        best_feature   = None
        best_threshold = None

        n_features = node.samples.shape[1]

        # Feature subsampling
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        elif self.max_features == 'sqrt':
            k = max(1, int(np.sqrt(n_features)))
            feature_indices = np.random.choice(n_features, size=k, replace=False)
        else:
            k = max(1, min(int(self.max_features), n_features))
            feature_indices = np.random.choice(n_features, size=k, replace=False)

        for col in feature_indices:
            unique = np.sort(np.unique(node.samples[:, col]))
            if len(unique) > 20:
                thresholds = np.linspace(unique[0], unique[-1], num=20)
            else:
                thresholds = (unique[:-1] + unique[1:]) / 2  # midpoints

            for thresh in thresholds:
                left_mask  = node.samples[:, col] < thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                ig = self.information_gain(
                    node,
                    node.labels[left_mask],
                    node.labels[right_mask],
                )

                if ig > best_gain:
                    best_gain      = ig
                    best_threshold = thresh
                    best_feature   = col

        return best_feature, best_threshold


    def build_tree(self, node):
        # leaf
        if (
            node.depth >= self.max_depth
            or len(node.labels) <= self.min_samples_split
            or len(np.unique(node.labels)) == 1
        ):
            node.is_leaf = True
            node.pred    = np.bincount(node.labels).argmax()
            node.samples = None  
            return

        feature, threshold = self.best_split(node)
        if feature is None:
            node.is_leaf = True
            node.pred    = np.bincount(node.labels).argmax()
            return

        # Internal node — create children - recursive call
        mask = node.samples[:, feature] < threshold

        left_node  = Node()
        right_node = Node()

        left_node.samples  = node.samples[mask]
        left_node.labels   = node.labels[mask]
        right_node.samples = node.samples[~mask]
        right_node.labels  = node.labels[~mask]

        left_node.depth  = node.depth + 1
        right_node.depth = node.depth + 1

        left_node.entropy  = self.entropy(left_node.labels)
        right_node.entropy = self.entropy(right_node.labels)

        node.feature     = feature
        node.threshold   = threshold
        node.left_child  = left_node
        node.right_child = right_node
        node.samples     = None  

        self.build_tree(left_node)
        self.build_tree(right_node)


    def fit(self, X_train, y_train):
        self.root         = Node()
        self.root.depth   = 0
        self.root.samples = X_train
        self.root.labels  = y_train
        self.root.entropy = self.entropy(y_train)
        self.build_tree(self.root)

    def predict(self, X):
        predictions = np.empty(X.shape[0])
        for i, sample in enumerate(X):
            node = self.root
            while not node.is_leaf:
                if sample[node.feature] < node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
            predictions[i] = node.pred
        return predictions