from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.datasets import load_iris, load_breast_cancer, fetch_california_housing
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from check_inverse import check_inverse_rank

# helper: regularized regression
def linear_regression_train(X, y, lamda):
    XTX = X.T @ X 
    if (check_inverse_rank(X.T @ X)):
        print("Matrix invertible → no regularization")
        w = inv(X.T @ X) @ X.T @ y
    else:
        print("Matrix not invertible → apply regularization")
        w = inv(X.T @ X + lamda + np.eye(X.shape[1])) @ X.T @ y
    print(f"w: {w}")
    return w

# helper: prediction
def linear_regression_predict(X, W):
    print(f"predicted label: {np.argmax(X @ W, axis=1)+1}")
    return X @ W

# helper: classification post-processing
def classify_outputs(Y_pred, is_bin, threshold=0.5):
    if is_bin:
        return (Y_pred > threshold).astype(int)
    else:
        return np.argmax(Y_pred, axis=1)

# main workflow
def linear_regression_pipeline(X, y,
                               X_train, X_val, X_test,
                               y_train, y_val, y_test,
                               is_class, is_bin, is_multi,
                               order, lamda, threshold,
                               test_size, val_size, random_state,
                               use_feature_selection,
                               normalize):
    
    # handles data splitting
    if X_train is None or y_train is None:
        if val_size == 0 or val_size is None:
            # only split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            X_val, y_val = None, None
        else:
            # split into train, val, and test sets
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
    
    # feature selection pre-processing (if needed)
    if use_feature_selection:
        df = X_train.copy()
        df['target'] = y_train
        correlations = df.corr()['target'].drop('target')
        top2_features = correlations.abs().sort_values(ascending=False).head(2).index.tolist()
        X_train = X_train[top2_features]
        X_val = X_val[top2_features]
        X_test = X_test[top2_features]

    # normalization of data
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        if X_val is not None:
            X_val = scaler.transform(X_val)
        X_test = scaler.fit_transform(X_test)
    else:
        pass

    # feature expansion
    poly = PolynomialFeatures(order)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.fit_transform(X_val) if X_val is not None else None
    X_test_poly = poly.fit_transform(X_test)
    
    # label processing
    encoder = OneHotEncoder(sparse_output=False)
    if is_multi:
        y_train_proc = encoder.fit_transform(y_train.reshape(-1,1))
        if y_val is not None:
            y_val_proc = encoder.fit_transform(y_val.reshape(-1,1))
        y_test_proc = encoder.fit_transform(y_test.reshape(-1,1))
    else:
        y_train_proc, y_test_proc = y_train, y_test
        if y_val is not None:
            y_val_proc = y_val
    
    # train model
    W = linear_regression_train(X_train_poly, y_train_proc, lamda)
    
    # predictions
    Ytr_pred = linear_regression_predict(X_train_poly, W)
    Yval_pred = linear_regression_predict(X_val_poly, W) if X_val_poly is not None else None
    Yts_pred = linear_regression_predict(X_test_poly, W)
    
    # classificiation handling
    if is_class:
        if is_bin:
            Ytr_class = classify_outputs(Ytr_pred, is_bin, threshold)
            if y_val is not None:
                Yval_class = classify_outputs(Yval_pred, is_bin, threshold)
            Yts_class = classify_outputs(Yts_pred, is_bin, threshold)
        elif is_multi:
            Ytr_class = classify_outputs(Ytr_pred, is_bin, threshold)
            if y_val is not None:
                Yval_class = classify_outputs(Yval_pred, is_bin, threshold)
            Yts_class = classify_outputs(Yts_pred, is_bin, threshold)
    else:
        Ytr_class = Ytr_pred
        if y_val is not None:
            Yval_class = Yval_pred
        Yts_class = Yts_pred
    
    # evaluate performance
    if is_class:
        if is_multi:
            y_train_true = np.argmax(y_train_proc, axis=1)
            y_val_true   = np.argmax(y_val_proc, axis=1) if y_val is not None else None
            y_test_true  = np.argmax(y_test_proc, axis=1)
        elif is_bin:
            y_train_true = (y_train > threshold).astype(int)
            y_val_true   = (y_val > threshold).astype(int) if y_val is not None else None
            y_test_true  = (y_test > threshold).astype(int)
            
        print(f"Train Acc: {accuracy_score(y_train_true, Ytr_class):.3f}")
        if X_val is not None:
            print(f"Val Acc:   {accuracy_score(y_val_true, Yval_class):.3f}")
        print(f"Test Acc:  {accuracy_score(y_test_true, Yts_class):.3f}")
        print("Confusion matrix (Test):")
        print(confusion_matrix(y_test_true, Yts_class))  
    else:
        print(f"Train MSE: {mean_squared_error(y_train, Ytr_pred):.3f}")
        if X_val is not None:
            print(f"Val MSE:   {mean_squared_error(y_val, Yval_pred):.3f}")
        print(f"Test MSE:  {mean_squared_error(y_test, Yts_pred):.3f}")

data = load_iris()
X = data.data
y = data.target

# USAGE:
# change use_feature_selection part in main pipeline code (if needed)
# linear_regression_pipeline(X, y,
                            # X_train, X_val, X_test,
                            # y_train, y_val, y_test,
                            # is_class, is_bin, is_multi,
                            # order, lamda, threshold,
                            # test_size, val_size, random_state,
                            # use_feature_selection,
                            # normalize)

linear_regression_pipeline(X, y,
                           None, None, None,
                           None, None, None, 
                           True, False, True,
                           1, 0.00001, 0.0,
                           0.2, 0.5, 42,
                           False,
                           False)
                        