import numpy as np
import matplotlib.pyplot as plt

from dataset import load_data
from decision_tree import DecisionTree

seed = 10
np.random.seed(seed)


# ── Grid search (greedy=False — commented out) ────────────────────────────────
#
# max_depth_arr         = [5, 10, 15, 20]
# min_samples_split_arr = [2, 10, 20, 50, 100]
#
# def grid_search(X_train, X_test, y_train, y_test):
#     results = {}
#     for max_depth in max_depth_arr:
#         for min_samples_split in min_samples_split_arr:
#             tree = DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split)
#             tree.fit(X_train, y_train)
#             acc = np.mean(tree.predict(X_test) == y_test)
#             print(f"max_depth={max_depth:2d}, min_samples_split={min_samples_split:3d}  ->  {acc:.4f}")
#             results[f'{max_depth}, {min_samples_split}'] = acc
#
#     accuracy_matrix = np.zeros((len(max_depth_arr), len(min_samples_split_arr)))
#     for i, depth in enumerate(max_depth_arr):
#         for j, split in enumerate(min_samples_split_arr):
#             accuracy_matrix[i, j] = results[f'{depth}, {split}']
#
#     plt.figure()
#     plt.imshow(accuracy_matrix, cmap='viridis')
#     plt.colorbar(label='Accuracy')
#     plt.xticks(np.arange(len(min_samples_split_arr)), min_samples_split_arr)
#     plt.yticks(np.arange(len(max_depth_arr)), max_depth_arr)
#     plt.xlabel('min_samples_split')
#     plt.ylabel('max_depth')
#     plt.title('Decision Tree Hyperparameter Optimisation')
#     for i in range(len(max_depth_arr)):
#         for j in range(len(min_samples_split_arr)):
#             plt.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
#                      ha='center', va='center', color='white')
#     plt.tight_layout()
#     plt.savefig('dt_optimisation_heatmap.png')
#     print("Heatmap saved to dt_optimisation_heatmap.png")
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("Loading and preprocessing data ...")
    X_train, X_test, y_train, y_test = load_data()

    # ── Final model — best hyperparameters, greedy search disabled ────────────
    print("\n-- Decision Tree (max_depth=5, min_samples_split=2) --")
    tree = DecisionTree(max_depth=5, min_samples_split=2)
    tree.fit(X_train, y_train)
    acc = np.mean(tree.predict(X_test) == y_test)
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()