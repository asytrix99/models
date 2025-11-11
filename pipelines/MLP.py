import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_digits

def ReLU(z): return np.maximum(0, z)
def Sigmoid(z): return 1 / (1 + np.exp(-z))
def Softmax(z, eps=1e-15):
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return np.clip(exp_z / np.sum(exp_z, axis=1, keepdims=True), eps, 1 - eps)

def cross_entropy_cost(Y, Y_hat, eps=1e-15):
    Y_hat = np.clip(Y_hat, eps, 1 - eps)
    return -np.mean(np.sum(Y * np.log(Y_hat), axis=1))

def forward_pass_N(X, W_list, is_bin):
    A = X
    cache = []
    for i in range(len(W_list)-1):
        Z = A @ W_list[i]
        A_relu = ReLU(Z)
        A = np.hstack((np.ones((A_relu.shape[0],1)), A_relu))
        cache.append((Z, A))
    ZL = A @ W_list[-1]
    Y_hat = Sigmoid(ZL) if is_bin else Softmax(ZL)
    return Y_hat, cache

def backward_pass_N(X, Y, Y_hat, W_list, cache, lr, lamda, is_bin):
    N = Y.shape[0]
    E = (Y_hat - Y) / N
    grads = []
    for i in reversed(range(len(W_list))):
        Z_prev, A_prev = cache[i-1] if i > 0 else (None, X)
        A_input = A_prev if i == 0 else np.hstack((np.ones((A_prev.shape[0],1)), ReLU(Z_prev)))
        G = A_input.T @ E + lamda * W_list[i]
        grads.insert(0, G)
        if i > 0:
            E = (E @ W_list[i][1:].T) * (Z_prev > 0)
    for i in range(len(W_list)):
        W_list[i] -= lr * grads[i]
    return W_list


def train_MLP_Nlayer(X, Y, W_list, lr, lamda, num_iters, is_bin):
    cost_vec = np.zeros(num_iters)
    for i in range(num_iters):
        Y_hat, cache = forward_pass_N(X, W_list, is_bin)
        cost = cross_entropy_cost(Y, Y_hat)
        cost_vec[i] = cost
        W_list = backward_pass_N(X, Y, Y_hat, W_list, cache, lr, lamda, is_bin)
        if i % 1000 == 0: print(f"Iter {i}, Cost: {cost:.5f}")
    return W_list, cost_vec

def MLP_Nlayer_pipeline(
    X, y,
    is_bin, is_multi,
    poly_order, threshold, lamda, learning_rates, num_iters,
    hidden_layer_configs,
    test_size, val_size, random_state, seed,
    use_feature_selection, 
    normalize):

    encoder = OneHotEncoder(sparse_output=False)
    if is_multi: y = encoder.fit_transform(y.reshape(-1,1))
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)

    if use_feature_selection and is_bin:
        import pandas as pd
        df = pd.DataFrame(X_train).copy()
        df['target'] = y_train
        cor = df.corr()['target'].drop('target')
        filt = cor[cor.abs()>0.5].index.tolist()
        keep = []
        cm = df[filt].corr().abs()
        for f in filt:
            if all(cm.loc[f,k]<=0.9 for k in keep): keep.append(f)
        X_train, X_val, X_test = X_train[:,keep], X_val[:,keep], X_test[:,keep]

    if normalize:
        scaler = StandardScaler()
        X_train, X_val, X_test = scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    Poly = PolynomialFeatures(poly_order)
    X_train, X_val, X_test = Poly.fit_transform(X_train), Poly.transform(X_val), Poly.transform(X_test)

    if is_bin and y_train.ndim==1:
        y_train, y_val, y_test = y_train.reshape(-1,1), y_val.reshape(-1,1), y_test.reshape(-1,1)
    if is_bin and not is_multi:
        Y_train, Y_val, Y_test = y_train, y_val, y_test
    else:
        Y_train, Y_val, Y_test = y_train, y_val, y_test

    best_val_acc, best_conf = -np.inf, None
    np.random.seed(seed)
    curves = {}

    for lr in learning_rates:
        for hidden_layers in hidden_layer_configs:
            layer_dims = [X_train.shape[1]] + hidden_layers + [Y_train.shape[1] if not is_bin else 1]
            W_list = []
            for i in range(len(layer_dims) - 1):
                in_dim = layer_dims[i] + (0 if i == 0 else 1)
                W_list.append(np.random.randn(in_dim, layer_dims[i+1]))
            W_trained, cost_vec = train_MLP_Nlayer(X_train, Y_train, W_list, lr, lamda, num_iters, is_bin)
            curves[(lr, tuple(hidden_layers))] = cost_vec
            Y_val_hat, _ = forward_pass_N(X_val, W_trained, is_bin)
            Y_val_cls = (Y_val_hat>threshold).astype(int) if is_bin else np.argmax(Y_val_hat,1)
            y_val_true = Y_val.flatten() if is_bin else np.argmax(Y_val,1)
            val_acc = accuracy_score(y_val_true, Y_val_cls)
            if val_acc > best_val_acc:
                best_val_acc, best_conf, best_W, best_cfg = val_acc, confusion_matrix(y_val_true, Y_val_cls), W_trained, (lr, hidden_layers)

    plt.figure(figsize=(8,6))
    for (lr, hl), cv in curves.items():
        plt.plot(cv, label=f'LR={lr}, HL={hl}')
    plt.xlabel('Iterations'); plt.ylabel('Cost'); plt.title('MLP Loss Curves'); plt.legend(); plt.show()

    print("\n✅ Best Model:")
    print(f"Learning Rate: {best_cfg[0]}, Hidden Layers: {best_cfg[1]}, Val Acc: {best_val_acc:.4f}")
    print("Confusion Matrix (Val Set):\n", best_conf)

data = load_digits()
X, y = data.data, data.target

# USAGE
# def MLP_Nlayer_pipeline(
#     X, y,
#     is_bin, is_multi,
#     poly_order, threshold, lamda, learning_rates, num_iters,
#     hidden_layer_configs,
#     test_size, val_size, random_state, seed,
#     use_feature_selection, 
#     normalize):
    
MLP_Nlayer_pipeline(
    X, y,
    False, True,
    1, 0.5, 0.0, [0.01], 20000,
    [32,32,32], 
    0.3, 0.5, 42, 42,
    False,
    True
)

