import os
import zipfile
import numpy as np
import pandas as pd
import cv2
from skimage.feature import hog

from Exercise1_2 import Ex01_2

# extract dataset zip file to ./data/dataset
def extract_dataset(zip_path="./dataset.zip", output_dir="./data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_dir)



def load_annotations(base_path="./data"):
    df = pd.read_csv(os.path.join(base_path, "dataset/annotations.csv"))

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    return train, val, test


def add_bbox_features(df, image_dir):
    widths, heights = [], []

    for f in df["filename"]:
        img = cv2.imread(os.path.join(image_dir, f))
        h, w, _ = img.shape
        widths.append(w)
        heights.append(h)

    df["original_width"] = widths
    df["original_height"] = heights

    df["x_center"] = ((df["bbox_xmin"] + df["bbox_xmax"]) / 2) / df["original_width"]
    df["y_center"] = ((df["bbox_ymin"] + df["bbox_ymax"]) / 2) / df["original_height"]

    df["x_normal"] = (df["bbox_xmax"] - df["bbox_xmin"]) / df["original_width"]
    df["y_normal"] = (df["bbox_ymax"] - df["bbox_ymin"]) / df["original_height"]

    return df.reset_index(drop=True)



def extract_hog_features(df, image_dir):
    features = []

    for f in df["filename"]:
        img = cv2.imread(os.path.join(image_dir, f))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))

        # histogram of oriented gradients with 9 bins for each 8x8 cell
        feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2)
        )
        features.append(feat)

    return np.array(features)


# Normalization
def zscore_params(X_train):
    mu    = X_train.mean(axis=0)          
    sigma = X_train.std(axis=0) + 1e-8                                    
    return mu, sigma
 
def zscore(X, mu, sigma) :
    return (X - mu) / sigma
##################################

#datapipeline function from excercise 1 - we just need species
def run_datapipeline(zip_path="./dataset.zip"):
    extract_dataset(zip_path)

    base_path = "./data"
    image_dir = os.path.join(base_path, "dataset/images")

    train_df, val_df, test_df = load_annotations(base_path)

    train_df = add_bbox_features(train_df, image_dir)
    val_df = add_bbox_features(val_df, image_dir)
    test_df = add_bbox_features(test_df, image_dir)

    X_train = extract_hog_features(train_df, image_dir)
    X_val = extract_hog_features(val_df, image_dir)
    X_test = extract_hog_features(test_df, image_dir)

    # to avoid the concentration of model on features with larger values but less importance
    mu, sigma = zscore_params(X_train)
    X_train = zscore(X_train, mu, sigma)
    X_val = zscore(X_val, mu, sigma)
    X_test = zscore(X_test, mu, sigma)

    # y_train = train_df[["x_center", "y_center", "x_normal", "y_normal"]].values
    # y_val = val_df[["x_center", "y_center", "x_normal", "y_normal"]].values
    # y_test = test_df[["x_center", "y_center", "x_normal", "y_normal"]].values

    y_train = (train_df["species"] == "dog").astype(int).values
    y_val   = (val_df["species"] == "dog").astype(int).values
    y_test  = (test_df["species"] == "dog").astype(int).values
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }



if __name__ == "__main__":
    
    data = run_datapipeline()

    X_train, y_train = data["train"]
    X_val, y_val = data["val"]  
    X_test, y_test = data["test"]
    
    # Question 2.1
    
    # model = Ex01_2(gamma=0.01, lr=1e-2, epochs=1000)
    # model.train(X_train, y_train, X_val, y_val)
    # gammas = [1e-3, 1e-2, 1e-1, 1]
    gammas = [None]
    # lrs = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    lrs = [1e-2, 1e-3]
    # epochs_list = [200, 500, 1000, 2000, 5000, 10000, 12000, 15000, 20000, 30000]
    epochs_list = [1000000]

    # best_model = None
    # best_val_loss = float('inf')
    # best_params = None
    # for gamma in gammas:
    #     for lr in lrs:
    #         for epochs in epochs_list:

    #             print("\n" + "="*50)
    #             print(f"Training model: lr={lr}, epochs={epochs}")
    #             print("="*50)

    #             model = Ex01_2(gamma=gamma, lr=lr, epochs=epochs, earlty_stopping_patience=200)

    #             model.train(X_train, y_train, X_val, y_val)

    #             val_loss = model.best_val_loss 
    #             epoch_of_best = model.best_epoch
    #             print(f"Final Val Loss: {val_loss:.6f}")

    #             if val_loss < best_val_loss:
    #                 best_val_loss = val_loss
    #                 best_model = model
    #                 best_params = (lr, epochs)
    #                 best_epoch = epoch_of_best
    #                 gamma_of_best = model.gamma
    #             model.plot_losses()

    # print("\n" + "#"*50)
    # print("BEST MODEL FOUND")
    # print(f"LR: {best_params[0]}, Epochs: {best_params[1]}")
    # print(f"Best Val Loss: {best_val_loss:.6f}")
    # print(f"Best Epoch: {best_epoch}")
    # print(f"Best Gamma: {gamma_of_best}")
    # print("#"*50)

    # if gamma = none, heuristic method (for normalized data))
    # without normalization around 1e-3 is good gamma. 
    # learning rate = 1e-2 and number of epochs around 530000
    model = Ex01_2(gamma=None, lr=1e-2, epochs=536000, earlty_stopping_patience=200)
    model.train(X_train, y_train, X_val, y_val)
    model.plot_losses()
    
    # Question 2.2
    model.evaluate(X_test, y_test, X_train)