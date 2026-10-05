import os
import zipfile
import pandas as pd
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from scipy.optimize import minimize

#..........Data loading and Feature extraction........

IMG_DIR = os.path.join('./data', 'dataset', 'images')
CSV_PATH = os.path.join('./data', 'dataset', 'annotations.csv')

if not os.path.exists('./data'):
    os.makedirs('./data')

if os.path.exists('./dataset.zip'):
    with zipfile.ZipFile('./dataset.zip', 'r') as r:
        r.extractall('./data')


annots_df = pd.read_csv('./data/dataset/annotations.csv')

def feature_extraction(df):
    features_list = []
    labels = []
    
    for _, item in df.iterrows():
        img_path = os.path.join('./data/dataset/images', item['filename'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (128, 128)) 
        feat = hog(gray_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        features_list.append(feat)
        labels.append([item['bbox_xmin'] / w, item['bbox_ymin'] / h, item['bbox_xmax'] / w, item['bbox_ymax'] / h ])
    return np.array(features_list), np.array(labels)

train_df= annots_df[annots_df['split'] == 'train']
val_df = annots_df[annots_df['split'] == 'val']
test_df = annots_df[annots_df['split'] == 'test']

X_train, y_train = feature_extraction(train_df)
X_val, y_val = feature_extraction(val_df)
X_test, y_test = feature_extraction(test_df)
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

n_components = 150
for n in [50, 100, 150, 200, 250, 300]:
    pca_tmp = PCA(n_components=n)
    pca_tmp.fit(X_train)
    print(f"  n={n:3d}  variance explained: {pca_tmp.explained_variance_ratio_.sum():.2%}")
    if pca_tmp.explained_variance_ratio_.sum() > 0.75:
        n_components = n
        break
print(f'best_n: {n_components}')
pca = PCA(n_components)
X_train = pca.fit_transform(X_train)
X_val   = pca.transform(X_val)
X_test  = pca.transform(X_test)
print(f"after PCA: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

#...............mse.............................

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#..............IOU..............................

def calculate_iou(y_true, y_pred):
    xmin = np.maximum(y_true[:, 0], y_pred[:, 0])
    ymin = np.maximum(y_true[:, 1], y_pred[:, 1])
    xmax = np.minimum(y_true[:, 2], y_pred[:, 2])
    ymax = np.minimum(y_true[:, 3], y_pred[:, 3])
    w = np.maximum(0, xmax - xmin)
    h = np.maximum(0, ymax - ymin)
    intersection = w * h
    area_true = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    area_pred = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    union = area_true + area_pred - intersection
    iou = intersection / np.maximum(union, 1e-8)
    return np.mean(iou)

#...............Linear regression................

def train_linear_reg(x, y):
        x = np.concatenate((np.ones((len(x),1)), x) , axis=1)
        a = x.T @ x
        b = x.T @ y
        w = np.linalg.solve(a, b)
        return w
        
def predict_linear_reg(x, w):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = np.concatenate((np.ones((len(x), 1)), x), axis=1)  
    pred = x @ w
    return pred

w_linear_reg = train_linear_reg(X_train, y_train)
print("w shape:", w_linear_reg.shape)
y_val_pred = predict_linear_reg(X_val, w_linear_reg)
mse_linear_reg = calculate_mse(y_val, y_val_pred)
iou_linear_reg = calculate_iou(y_val, y_val_pred)
print(f'mse linear regression validation: {mse_linear_reg}')

#................GPR.............................

def linear_kernel(x1, x2):
    return (x1 @ x2.T) + 1

def polynomial_kernel(x1, x2, p=2):
    dot = (x1 @ x2.T) + 1
    return np.power(np.maximum(dot, 1e-9), p)

def rbf_kernel(x1, x2, l=0.1):
    a = np.sum(x1**2, axis=1).reshape(-1,1)
    b = np.sum(x2**2, axis=1).reshape(1,-1)
    c = a + b - 2 * x1 @ x2.T
    return np.exp(-0.5 * (c / (l**2)))

def train_gpr(x, y, kernel, ratio_variance=1e-5, p=2, l=0.1):
    if kernel == 'linear':
        K = linear_kernel(x, x)
    elif kernel == 'poly':
        K = polynomial_kernel(x, x, p)
    elif kernel == 'rbf':
        K = rbf_kernel(x, x, l)
  
    n = K.shape[0]
    K += ratio_variance * np.eye(n)
    K_inverse = np.linalg.pinv(K)
    w_xmin = K_inverse @ y[:, 0]
    w_ymin = K_inverse @ y[:, 1]
    w_xmax = K_inverse @ y[:, 2]
    w_ymax = K_inverse @ y[:, 3]
    return (w_xmin, w_ymin, w_xmax, w_ymax)

def predict_gpr(x, x_test, weights, kernel, **kwargs):
    w_xmin, w_ymin, w_xmax, w_ymax = weights
    if kernel == 'linear':
        k_star = linear_kernel(x_test, x)
    elif kernel == 'poly':
         k_star = polynomial_kernel(x_test, x, **kwargs)
    elif kernel == 'rbf':
         k_star = rbf_kernel(x_test, x, **kwargs)

    p_xmin = k_star @ w_xmin
    p_ymin = k_star @ w_ymin
    p_xmax = k_star @ w_xmax
    p_ymax = k_star @ w_ymax
    preds = np.stack([p_xmin, p_ymin, p_xmax, p_ymax], axis=1)
    return preds

#..................optimization.........................

#log_max_likelihood = -1/2 * y^T * Ky^-1 * y - 1/2 * log|Ky|- n/2 * log(2*pi)
def likelihood(hyperparam, x, y, kernel_type):
    try:
        n = x.shape[0]
        params = np.exp(hyperparam)
        noise_var = params[-1]

        if kernel_type == 'linear':
            K = linear_kernel(x, x)

        elif kernel_type == 'poly':
            p = params[0]
            K = polynomial_kernel(x, x, p)
        
        elif kernel_type == 'rbf':
            l= params[0]
            K = rbf_kernel(x, x, l)

        Ky = K + (noise_var + 1e-7) * np.eye(n)
    
        L = np.linalg.cholesky(Ky)
        alpha = np.linalg.solve(Ky, y)
        A = -0.5 * np.sum(y * alpha)
        logdet = 2 * np.sum(np.log(np.diag(L)))
        B = -0.5 * 4 * logdet
        C = -0.5 * n * 4 * np.log(2 * np.pi)
        lml = A + B + C
        return -lml 
    except np.linalg.LinAlgError:
        return 1e10
 
        
def optimization(x, y, kernel_type):
    if kernel_type == 'linear':
        initial_log_params = np.log([1e-3])
        bound = [(-10, 5)]
    elif kernel_type == 'poly':
        initial_log_params = np.log([2.0, 1e-3])
        bound = [(-5, 5), (-10, 5)]
    elif kernel_type == 'rbf':
        initial_log_params = np.log([1.0, 1e-3])
        bound = [(-5, 5), (-10, 5)]

    res = minimize(likelihood, initial_log_params, args=(x, y, kernel_type),method='L-BFGS-B', bounds=bound)
    best_params = np.exp(res.x)
    print(f"Best parameters for {kernel_type}: {best_params}")
    return best_params


#............linear kernel.............

best_linear_param = optimization(X_train, y_train, 'linear')
noise_var = best_linear_param[-1]
w_linear_kernel = train_gpr(X_train, y_train, 'linear', ratio_variance=noise_var)
val_pred_linear_k = predict_gpr(X_train, X_val, w_linear_kernel, 'linear')
print(f"Linear kernel val MSE: {calculate_mse(y_val, val_pred_linear_k):.6f}")    
print(f"Linear kernel val IOU: {calculate_iou(y_val, val_pred_linear_k):.6f}")    

#............polynomial kernel............

best_poly_param = optimization(X_train, y_train, 'poly')
noise_var = best_poly_param[-1]
p_best = best_poly_param[0]
w_poly_kernel = train_gpr(X_train, y_train, 'poly', ratio_variance=noise_var, p=p_best)
val_pred_poly = predict_gpr(X_train, X_val, w_poly_kernel, 'poly', p=p_best)
print(f"Poly kernel Val MSE: {calculate_mse(y_val, val_pred_poly):.6f}")
print(f"Poly kernel Val IOU: {calculate_iou(y_val, val_pred_poly):.6f}")

#............rbf kernel...................


best_rbf_param = optimization(X_train, y_train, 'rbf')
noise_var = best_rbf_param[-1]
l_best = best_rbf_param[0]
w_rbf_kernel = train_gpr(X_train, y_train, 'rbf', ratio_variance=noise_var, l=l_best)
val_pred_rbf = predict_gpr(X_train, X_val, w_rbf_kernel, 'rbf', l=l_best)
print(f"RBF kernel val MSE: {calculate_mse(y_val, val_pred_rbf):.6f}")
print(f"RBF kernel val IOU: {calculate_iou(y_val, val_pred_rbf):.6f}")


#.............test data..................

y_test_pred_reg  = predict_linear_reg(X_test, w_linear_reg)
mse = calculate_mse(y_test, y_test_pred_reg)
iou = calculate_iou(y_test, y_test_pred_reg)
print(f"Linear regression test:   MSE={mse:.6f}  IoU={iou:.4f}")

y_test_pred_linear = predict_gpr(X_train, X_test, w_linear_kernel, 'linear')
mse = calculate_mse(y_test, y_test_pred_linear)
iou = calculate_iou(y_test, y_test_pred_linear)
print(f"Lineare kernle test:   MSE={mse:.6f}  IoU={iou:.4f}")

y_test_pred_poly = predict_gpr(X_train, X_test, w_poly_kernel, 'poly', p=p_best)
mse = calculate_mse(y_test, y_test_pred_poly)
iou = calculate_iou(y_test, y_test_pred_poly)
print(f"poly kernle test:   MSE={mse:.6f}  IoU={iou:.4f}")

y_test_pred_rbf  = predict_gpr(X_train, X_test, w_rbf_kernel,  'rbf',  l=l_best)
mse = calculate_mse(y_test, y_test_pred_rbf)
iou = calculate_iou(y_test, y_test_pred_rbf)
print(f"rbf kernle test:   MSE={mse:.6f}  IoU={iou:.4f}")

#..............comparing result................

'''The linear regression and linear kernel produce same and 
   accaptable results, showing that the relation in this task
   for bounding box coordinates and HOG and PCA features is somehow linear. 
   The polynomial kernel performs worse, maybe due to overfitting in 
   high-dimensional feature space. 
   The RBF kernel achieves the best performance, as it can captures non-linear 
   relations and is more robust to high-dimensional feature representations'''


#...............visualization..................

def visualize_predictions(model_name, y_true, y_pred, df, n=5):
    df = df.reset_index(drop=True)
    indices = random.sample(range(len(df)), n)
    _, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, idx in zip(axes, indices):
        row = df.iloc[idx]
        img = cv2.imread(os.path.join('./data/dataset/images', row['filename']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        ax.imshow(img)
        gt = y_true[idx]
        pr = y_pred[idx]
        rect_gt = patches.Rectangle(
            (gt[0]*w, gt[1]*h), (gt[2]-gt[0])*w, (gt[3]-gt[1])*h,
            linewidth=2, edgecolor='green', facecolor='none')
        rect_pr = patches.Rectangle(
            (pr[0]*w, pr[1]*h), (pr[2]-pr[0])*w, (pr[3]-pr[1])*h,
            linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect_gt)
        ax.add_patch(rect_pr)
        ax.set_title(f"Pred: {np.round(pr,3)}\nTruth: {np.round(gt,3)}", fontsize=6)
        ax.axis('off')
    plt.suptitle(model_name)
    plt.tight_layout()
    plt.show()

visualize_predictions('Linear_Reg',  y_test, y_test_pred_reg,  test_df)
visualize_predictions('GPR_Linear',  y_test, y_test_pred_linear, test_df)
visualize_predictions('GPR_Poly',    y_test, y_test_pred_poly, test_df)
visualize_predictions('GPR_RBF',     y_test, y_test_pred_rbf,  test_df)