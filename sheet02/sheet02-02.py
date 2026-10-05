import numpy as np
import itertools
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from dataset import load_data
from random_forest import RandomForest

seed = 10



def _evaluate_combo(args):
    n_est, depth, feats, mss, X_train, X_test, y_train, y_test, worker_seed = args
    np.random.seed(worker_seed)
    rf = RandomForest(
        n_estimators=n_est,
        max_depth=depth,
        max_features=feats,
        min_samples_split=mss,
    )
    rf.fit(X_train, y_train)
    acc = np.mean(rf.predict(X_test) == y_test)
    return (n_est, depth, feats, mss, acc)


# ── Grid search ───────────────────────────────────────────────────────────────

def grid_search(X_train, X_test, y_train, y_test):
    n_estimators_arr      = [50, 100, 150]
    max_depth_arr         = [5, 10, 15, 20]
    n_features            = X_train.shape[1]
    max_features_arr      = list(range(1, n_features + 1))  # 1, 2, ..., n_features
    min_samples_split_arr = [2, 10, 20]

    combos = list(itertools.product(
        n_estimators_arr, max_depth_arr, max_features_arr, min_samples_split_arr
    ))

    args_list = [
        (n_est, depth, feats, mss, X_train, X_test, y_train, y_test, seed + i)
        for i, (n_est, depth, feats, mss) in enumerate(combos)
    ]

    n_workers = cpu_count()
    print(f"Running grid search over {len(combos)} combinations "
          f"using {n_workers} CPU cores ...\n")

    with Pool(processes=n_workers) as pool:
        raw_results = pool.map(_evaluate_combo, args_list)

    results     = {}
    best_acc    = 0
    best_params = {}

    for n_est, depth, feats, mss, acc in sorted(raw_results, key=lambda x: x[4], reverse=True):
        key = (n_est, depth, feats, mss)
        results[key] = acc
        print(f"n_est={n_est:3d}, depth={depth:2d}, feats={str(feats):4s}, mss={mss:3d}  ->  {acc:.4f}")
        if acc > best_acc:
            best_acc    = acc
            best_params = dict(
                n_estimators=n_est, max_depth=depth,
                max_features=feats, min_samples_split=mss,
            )

    print(f"\nBest params : {best_params}")
    print(f"Best acc    : {best_acc:.4f}")

    best_feats = best_params['max_features']
    mss_fix    = 2
    matrix1    = np.zeros((len(max_depth_arr), len(n_estimators_arr)))
    for i, depth in enumerate(max_depth_arr):
        for j, n_est in enumerate(n_estimators_arr):
            matrix1[i, j] = results.get((n_est, depth, best_feats, mss_fix), 0)

    plt.figure()
    plt.imshow(matrix1, cmap='viridis')
    plt.colorbar(label='Accuracy')
    plt.xticks(np.arange(len(n_estimators_arr)), n_estimators_arr)
    plt.yticks(np.arange(len(max_depth_arr)), max_depth_arr)
    plt.xlabel('n_estimators')
    plt.ylabel('max_depth')
    plt.title(f"RF Grid Search (max_features={best_feats}, min_samples_split={mss_fix})")
    for i in range(len(max_depth_arr)):
        for j in range(len(n_estimators_arr)):
            plt.text(j, i, f'{matrix1[i, j]:.3f}',
                     ha='center', va='center', color='white')
    plt.tight_layout()
    plt.savefig('rf_heatmap_n_est_vs_depth.png')
    print("Heatmap saved to rf_heatmap_n_est_vs_depth.png")

    best_n_est = best_params['n_estimators']
    matrix2    = np.zeros((len(max_depth_arr), len(max_features_arr)))
    for i, depth in enumerate(max_depth_arr):
        for j, feats in enumerate(max_features_arr):
            matrix2[i, j] = results.get((best_n_est, depth, feats, mss_fix), 0)

    plt.figure(figsize=(max(8, n_features), 4))
    plt.imshow(matrix2, cmap='viridis', aspect='auto')
    plt.colorbar(label='Accuracy')
    plt.xticks(np.arange(len(max_features_arr)), max_features_arr)
    plt.yticks(np.arange(len(max_depth_arr)), max_depth_arr)
    plt.xlabel('max_features')
    plt.ylabel('max_depth')
    plt.title(f"RF Grid Search (n_estimators={best_n_est}, min_samples_split={mss_fix})")
    for i in range(len(max_depth_arr)):
        for j in range(len(max_features_arr)):
            plt.text(j, i, f'{matrix2[i, j]:.3f}',
                     ha='center', va='center', color='white', fontsize=7)
    plt.tight_layout()
    plt.savefig('rf_heatmap_features_vs_depth.png')
    print("Heatmap saved to rf_heatmap_features_vs_depth.png")

    return best_params



def main():
    print("Loading and preprocessing data ...")
    X_train, X_test, y_train, y_test = load_data()

    # best_params = grid_search(X_train, X_test, y_train, y_test)
    best_params = dict(
            n_estimators=50, max_depth=10,
            max_features=4, min_samples_split=2,
        )  # from a previous run to save time
    np.random.seed(seed)
    rf = RandomForest(**best_params)
    rf.fit(X_train, y_train)
    acc = np.mean(rf.predict(X_test) == y_test)
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()