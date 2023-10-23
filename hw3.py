from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


# done
def load_and_center_dataset(filename):
    data = np.load(filename)
    result = data - np.mean(data, 0)
    return result


# done
def get_covariance(dataset):
    return np.dot(np.transpose(dataset), dataset) / (len(dataset) - 1)


# done
def get_eig(S, m):
    w, v = eigh(S, subset_by_index=[len(S) - m, len(S) - 1])
    return np.diag(np.flip(w)), np.fliplr(v)


# done
def get_eig_prop(S, prop):
    w, v = eigh(S, subset_by_value=[S.trace() * prop, np.inf])
    return np.diag(np.flip(w)), np.fliplr(v)


# done
def project_image(image, U):
    return np.ndarray.flatten(U @ (U.T @ image))


# done
def display_image(orig, proj):
    orig = np.reshape(orig, (32, 32)).T
    proj = np.reshape(proj, (32, 32)).T
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    ax2.set_title('Projection')
    fig.colorbar(ax1.imshow(orig, aspect='equal'), ax=ax1, shrink=0.5)
    fig.colorbar(ax2.imshow(proj, aspect='equal'), ax=ax2, shrink=0.5)
    plt.show()
