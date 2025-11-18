from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.datasets import fetch_openml, load_iris, load_breast_cancer, fetch_california_housing
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
    return X @ W

# helper: classification post-processing
def classify_outputs(Y_pred, is_bin, threshold=0.5):
    if is_bin:
        return (Y_pred > threshold).astype(int)
    else:
        return np.argmax(Y_pred, axis=1)

# main workflow
def polynomial_order_sweep(X, y,
                           X_train, X_val, X_test,
                           y_train, y_val, y_test,
                           is_class, is_bin, is_multi,
                           max_order, lamda, threshold,
                           test_size, val_size, random_state,
                           use_feature_selection,
                           normalize):
    
    train_metric_list = []
    val_metric_list = []
    test_metric_list = []
    
    for order in range(1, max_order + 1):
        print("\n" + "="*50)
        print(f"🏗️  Running polynomial order: {order}")
        print("="*50)
       # handles data splitting
        if X_train is None or y_train is None:
            if test_size == 0 or test_size is None:
                # vanilla mode → no split, use full data for both training and testing
                print("Vanilla mode: training and testing on the same dataset.")
                X_train, y_train = X, y
                X_test, y_test = X, y
                X_val, y_val = None, None
            elif val_size == 0 or val_size is None:
                # only split into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                X_val, y_val = None, None
            else:
                # split into train, validation, and test sets
                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
        
        # feature selection pre-processing (if needed)
        if use_feature_selection:
            df = X_train
            df['target'] = y_train
            feature_names = df.corr()['target'].drop('target').sort_values(ascending=False).abs().head(2).index.tolist()
            # Subset the DataFrame to only include the selected features.
            X_train = X_train[feature_names]
            X_val = X_val[feature_names]
            X_test = X_test[feature_names]
                
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            if X_val is not None:
                X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

        # feature expansion
        poly = PolynomialFeatures(order)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.fit_transform(X_val) if X_val is not None else None
        X_test_poly = poly.fit_transform(X_test)
        
        # label processing
        encoder = OneHotEncoder(sparse_output=False)
        if is_multi:
            y_train_proc = encoder.fit_transform(y_train.reshape(-1,1))
            y_val_proc = encoder.fit_transform(y_val.reshape(-1,1)) if y_val is not None else None
            y_test_proc = encoder.fit_transform(y_test.reshape(-1,1))
        else:
            y_train_proc, y_test_proc = y_train, y_test
            if y_val is not None:
                y_val_proc = y_val
        
        # train model
        W = linear_regression_train(X_train_poly, y_train_proc, lamda)
        
        # predictions
        Ytr_pred = linear_regression_predict(X_train_poly, W)
        Yval_pred = linear_regression_predict(X_val_poly, W) if y_val is not None else None
        Yts_pred = linear_regression_predict(X_test_poly, W)
        
        # classificiation handling
        if is_class:
            if is_bin:
                Ytr_class = (Ytr_pred > threshold).astype(int)
                Yval_class = (Yval_pred > threshold).astype(int) if Yval_pred is not None else None
                Yts_class = (Yts_pred > threshold).astype(int)
                
                y_train_true = (y_train > threshold).astype(int)
                y_val_true   = (y_val > threshold).astype(int) if y_val is not None else None
                y_test_true  = (y_test > threshold).astype(int)
                
                metric_name = "Accuracy"
                train_metric = accuracy_score(y_train_true, Ytr_class)
                val_metric = accuracy_score(y_val_true, Yval_class) if y_val_true is not None else None
                test_metric = accuracy_score(y_test_true, Yts_class)
            elif is_multi:
                Ytr_class = np.argmax(Ytr_pred, axis=1)
                Yval_class = np.argmax(Yval_pred, axis=1) if Yval_pred is not None else None
                Yts_class = np.argmax(Yts_pred, axis=1)
                
                y_train_true = np.argmax(y_train_proc, axis=1)
                y_val_true = np.argmax(y_val_proc, axis=1) if y_val_proc is not None else None
                y_test_true = np.argmax(y_test_proc, axis=1)
                
                metric_name = "Accuracy"
                train_metric = accuracy_score(y_train_true, Ytr_class)
                val_metric = accuracy_score(y_val_true, Yval_class) if y_val_true is not None else None
                test_metric = accuracy_score(y_test_true, Yts_class)
            
            # adaptive printing depending on available sets
            if X_val is not None and X_test is not None:
                print(f"Order {order} → Train {metric_name}: {train_metric:.4f}, ",
                    f"Val {metric_name}: {val_metric:.4f}, Test {metric_name}: {test_metric:.4f}")
                print("Confusion matrix:")
                print(confusion_matrix(y_test_true, Yts_class))
            elif X_val is None and X_test is not None:
                print(f"Order {order} → Train {metric_name}: {train_metric:.4f}, ",
                    f"Test {metric_name}: {test_metric:.4f}")
                print("Confusion matrix:")
                print(confusion_matrix(y_test_true, Yts_class))
            elif X_val is None and X_test is None:
                print(f"Order {order} → Train {metric_name}: {train_metric:.4f}")
                print("Confusion matrix:")
                print(confusion_matrix(y_test_true, Yts_class))
                
        else:
            metric_name = "MSE"
            train_metric = mean_squared_error(y_train, Ytr_pred)
            val_metric = mean_squared_error(y_val, Yval_pred) if y_val is not None else None
            test_metric = mean_squared_error(y_test, Yts_pred)
            
            # 🔹 Add detailed per-order MSE printing here
            print(f"\n--- Polynomial Order {order} Results ---")
            print(f"Train MSE: {train_metric:.4f}")
            if val_metric is not None:
                print(f"Val   MSE: {val_metric:.4f}")
            print(f"Test  MSE: {test_metric:.4f}")
        
        train_metric_list.append(train_metric)
        val_metric_list.append(val_metric) if val_metric is not None else None
        test_metric_list.append(test_metric)
        
    plt.figure(figsize=(8,5))
    plt.plot(range(1, max_order + 1), train_metric_list, marker='o', label=f'Train {metric_name}')

    # Only plot validation metrics if available
    if X_val is not None and len(val_metric_list) > 0:
        plt.plot(range(1, max_order + 1), val_metric_list, marker='o', label=f'Val {metric_name}')

    # Only plot test metrics if available
    if X_test is not None and len(test_metric_list) > 0:
        plt.plot(range(1, max_order + 1), test_metric_list, marker='o', label=f'Test {metric_name}')

    plt.title(f'{metric_name} vs Polynomial Order (λ={lamda})')
    plt.xlabel('Polynomial Order')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.show()

    
    # Determine best polynomial order depending on available data
    if X_val is not None and len(val_metric_list) > 0:
        if metric_name == "MSE":
            best_order = np.argmin(val_metric_list) + 1
        else:
            best_order = np.argmax(val_metric_list) + 1
        print(f"\nBest polynomial order based on Validation {metric_name}: {best_order}")
        print(f"Validation {metric_name} at best order: {val_metric_list[best_order-1]:.4f}")
    elif X_test is not None and len(test_metric_list) > 0:
        # fallback to test set if no validation available
        if metric_name == "MSE":
            best_order = np.argmin(test_metric_list) + 1
        else:
            best_order = np.argmax(test_metric_list) + 1
        print(f"\nBest polynomial order based on Test {metric_name}: {best_order}")
        print(f"Test {metric_name} at best order: {test_metric_list[best_order-1]:.4f}")
    else:
        # vanilla mode: only training available
        if metric_name == "MSE":
            best_order = np.argmin(train_metric_list) + 1
        else:
            best_order = np.argmax(train_metric_list) + 1
        print(f"\nBest polynomial order based on Train {metric_name}: {best_order}")
        print(f"Train {metric_name} at best order: {train_metric_list[best_order-1]:.4f}")

        

data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target
# USAGE:
# change use_feature_selection part in main pipeline code (if needed)
# def polynomial_order_sweep(X, y,
                            # X_train, X_val, X_test,
                            # y_train, y_val, y_test,
                            # is_class, is_bin, is_multi,
                            # max_order, lamda, threshold,
                            # test_size, val_size, random_state,
                            # use_feature_selection,
                            # normalize)

polynomial_order_sweep(X, y,
                        None, None, None,
                        None, None, None, 
                        False, False, False,
                        6, 1, 0.0,
                        0.4, 0.5, 42,
                        True,
                        False)