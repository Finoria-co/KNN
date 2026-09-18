import numpy as np
import matplotlib.pyplot as plt

def make_blobs(n_classes=3, n_per_class=20, n_dims=2, spread=1.0, seed=42):
    rng = np.random.default_rng(seed)

    centers = rng.uniform(-5, 5, size=(n_classes, n_dims))

    X_list = []
    y_list = []
    for c in range(n_classes):
        pts = rng.standard_normal((n_per_class, n_dims)) * spread + centers[c]
        X_list.append(pts)
        y_list.append(np.full(n_per_class, c))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y

def knn_predict(X_train, y_train, X_query, k=5):
    
    X_train = np.asarray(X_train)
    X_query = np.asarray(X_query)

    diff = X_query[:, None, :] - X_train[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  
    k = min(k, X_train.shape[0])
    nn_idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]  # (M, k)

    n_classes = int(y_train.max()) + 1
    preds = np.empty(X_query.shape[0], dtype=int)

    for i in range(X_query.shape[0]):
        neighbor_labels = y_train[nn_idx[i]]
       
        votes = np.bincount(neighbor_labels, minlength=n_classes)
        preds[i] = np.argmax(votes)

    return preds


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

if __name__ == "__main__":
    N_CLASSES = 3
    N_DIMS = 2
    N_PER_CLASS = 3
    K = 2  

 
    X_train, y_train = make_blobs(
        n_classes=N_CLASSES, n_per_class=N_PER_CLASS, n_dims=N_DIMS, spread=0.5, seed=42
    )
    x_query = np.array([[0.0, 0.0]])

    diff = X_train - x_query
    distances = np.sqrt(np.sum(diff**2, axis=1))  
    nearest_indices = np.argsort(distances)[:K]   
    
    predicted_class = knn_predict(X_train, y_train, x_query, k=K)[0]

   
    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="viridis", edgecolor="k", s=80, label="Data Points")

    ax.scatter(x_query[0, 0], x_query[0, 1], c='red', marker='X', s=200, label=f"Target X (Predicted: Class {predicted_class})")

   
    ax.scatter(X_train[nearest_indices, 0], X_train[nearest_indices, 1],
               facecolors='none', edgecolors='red', s=250, linewidth=2, label=f"{K} Nearest Neighbors")

    ax.set_aspect('equal')  
    plt.title(f"K-Nearest Neighbor (K={K})")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()