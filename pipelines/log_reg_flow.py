import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
def compute_loss(y_true, y_pred, is_bin, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    if is_bin:
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss

def predict_proba(X, W, is_bin):
    z = X @ W
    if is_bin:
        return sigmoid(z)
    else:
        return softmax(z)

def predict_class(Y_pred, is_bin, threshold):
    if is_bin:
        return (Y_pred > threshold).astype(int)
    else:
        return np.argmax(Y_pred, axis=1)

def gradient_descent(X, y, is_bin, lr, num_iters, lamda, seed):
    n_samples, n_features = X.shape
    np.random.seed(seed)
    
    if is_bin:
        W = np.random.randn(n_features, 1)
    else:
        n_classes = y.shape[1]
        W = np.random.randn(n_features, n_classes)

    loss_list = []

    for i in range(num_iters):
        z = X @ W
        if is_bin:
            y_pred = sigmoid(z)
        else:
            y_pred = softmax(z)
            
        grad = (X.T @ (y_pred - y)) / n_samples + lamda * W
        W -= lr * grad
        loss = compute_loss(y, y_pred, is_bin)
        loss_list.append(loss)

        if i % 2000 == 0:
            print(f"Iteration {i}, Cost: {loss:.6f}")

    return W, loss_list

def logistic_order_lr_sweep(X, y,
                            X_train, X_val, X_test,
                            y_train, y_val, y_test,
                            is_bin, is_multi,
                            max_order, threshold, lamda, learning_rates, num_iters,
                            test_size, val_size, random_state, seed,
                            use_feature_selection,
                            normalize):
    
    encoder = OneHotEncoder(sparse_output=False)
    best_val_acc = -np.inf
    best_config = {}
    best_conf_mat = None
    lr_loss_curves = {}
    scaler = None
    
    # One-hot encode if multi-class
    if is_multi:
        y = encoder.fit_transform(y.reshape(-1, 1))
    
    # Split data once if not provided
    if X_train is None or y_train is None:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state
        )
    
    # Feature selection (do once before everything)
    if use_feature_selection:
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("Feature selection requires X to be a DataFrame")
        
        df = X_train.copy()
        df['target'] = y_train if is_bin else np.argmax(y_train, axis=1)
        corr = df.corr().drop('target').abs()
        filtered = corr[corr['target'] > 0.5].index.tolist()
        
        if filtered:
            corr_mat = df[filtered].corr().abs()
            selected_features = []
            for feature in filtered:
                if all(corr_mat[feature][selected] <= 0.9 for selected in selected_features):
                    selected_features.append(feature)
            
            X_train = X_train[selected_features]
            X_val = X_val[selected_features]
            X_test = X_test[selected_features]
            print(f"Selected {len(selected_features)} features: {selected_features}")
    
    # Convert to numpy if still DataFrame
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
        X_val = X_val.values
        X_test = X_test.values
    
    # Normalize once if needed
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
    # Reshape y for binary classification
    if is_bin and y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
    
    # Grid search over learning rates and polynomial orders
    for lr in learning_rates:
        train_acc_list = []
        val_acc_list = []
        
        for order in range(1, max_order + 1):
            print("\n" + "="*50)
            print(f"🏗️  Running LR={lr}, Polynomial Order={order}")
            print("="*50)
            
            # Create polynomial features
            poly = PolynomialFeatures(order)
            X_train_poly = poly.fit_transform(X_train)
            X_val_poly = poly.transform(X_val)
            X_test_poly = poly.transform(X_test)
            
            # Train model
            W, loss_curve = gradient_descent(
                X_train_poly, y_train, is_bin, lr, num_iters, lamda, seed
            )
            lr_loss_curves[(lr, order)] = loss_curve
            
            # Predictions
            Ytr_pred = predict_proba(X_train_poly, W, is_bin)
            Yval_pred = predict_proba(X_val_poly, W, is_bin)
            Yts_pred = predict_proba(X_test_poly, W, is_bin)

            Ytr_class = predict_class(Ytr_pred, is_bin, threshold)
            Yval_class = predict_class(Yval_pred, is_bin, threshold)
            Yts_class = predict_class(Yts_pred, is_bin, threshold)
            
            # Get true labels
            if is_multi:
                y_train_true = np.argmax(y_train, axis=1)
                y_val_true = np.argmax(y_val, axis=1)
                y_test_true = np.argmax(y_test, axis=1)
            else:
                y_train_true = y_train.flatten()
                y_val_true = y_val.flatten()
                y_test_true = y_test.flatten()
            
            # Calculate accuracies
            train_acc = accuracy_score(y_train_true, Ytr_class)
            val_acc = accuracy_score(y_val_true, Yval_class)
            test_acc = accuracy_score(y_test_true, Yts_class)
            
            print(f"\n--- Results for LR={lr}, Order={order} ---")
            print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            print("Confusion Matrix (Test Set):")
            print(confusion_matrix(y_test_true, Yts_class))

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_config = {
                    'lr': lr,
                    'order': order,
                    'W': W,
                    'poly': poly,
                    'scaler': scaler,
                    'test_acc': test_acc
                }
                best_conf_mat = confusion_matrix(y_test_true, Yts_class)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    for (lr, order), curve in lr_loss_curves.items():
        plt.plot(curve, label=f'LR={lr}, Order={order}', alpha=0.7)
    plt.title('Cost (Log Loss) vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*50)
    print("✅ Best Model Found:")
    print("="*50)
    print(f"Learning Rate: {best_config['lr']}")
    print(f"Polynomial Order: {best_config['order']}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {best_config['test_acc']:.4f}")
    print("\nConfusion Matrix (Test Set):")
    print(best_conf_mat)
    
    return best_config

# USAGE:
# logistic_order_lr_sweep(X, y,
#                             X_train, X_val, X_test,
#                             y_train, y_val, y_test,
#                             is_bin, is_multi,
#                             max_order, threshold, lamda, learning_rates, num_iters,
#                             test_size, val_size, random_state, seed,
#                             use_feature_selection,
#                             normalize)

data = load_iris()
X = data.data
y = data.target

best_model = logistic_order_lr_sweep(
    X, y,
    None, None, None,
    None, None, None,
    False, True,
    1, 0.5, 0.00001, [0.1, 0.01, 0.001], 20000,
    0.2, 0.5, 42, 42,
    False,
    False
)