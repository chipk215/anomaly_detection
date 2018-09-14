import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def read_matlab_data(file_name):
    data = spio.loadmat(file_name)
    return data


def plot_distributions(X, labels):
    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.tight_layout()
    axes[0].hist(X[:, 0], density=False)
    axes[0].set_title(labels[0])
    axes[1].hist( X[:, 1], density=False)
    axes[1].set_title(labels[1])
    axes[2].scatter(X[:,0],X[:,1])
    axes[2].set_title(labels[0] + " vs " + labels[1])
    plt.show()


def estimate_gaussian(X):
    """
    :param X: Matrix with samples in each row, features in the columns
    :return:
    """
    mu = np.transpose(np.mean(X, axis=0))
    sigma_sqr = np.transpose(np.var(X, axis=0))
    return mu, sigma_sqr


def multivariate_gaussian_v1(X, mu, sigma_sqr) :
    k = mu.size
    if (sigma_sqr.ndim == 1):
        sigma2 = np.diag(sigma_sqr)
    else:
        sigma2 = sigma_sqr
    # compute determinate of covariance matrix
    det_sigma2 = np.linalg.det(sigma2)
    multi_coeff = np.power(2 * np.pi, -k/2) * np.power(det_sigma2, -.5)
    # subtract the mean from each sample
    diff = X - mu
    sigma_inverse = np.linalg.pinv(sigma2)
    temp = np.matmul(diff, sigma_inverse) * diff
    sum_term = np.sum(temp, axis=1)
    exp_term = np.exp(-0.5 * sum_term)
    p = multi_coeff * exp_term
    return p


def select_threshold(y_val, p_val):
    def compute_precision(true_positives, false_positives):
        if (true_positives + false_positives) != 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
        return precision

    def compute_recall(true_positives, false_negatives):
        if (true_positives + false_negatives) != 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0
        return recall

    def compute_f1(precision, recall):
        if (precision + recall) != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        return f1

    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    best_f1_precision = 0
    best_f1_recall = 0
    p_min = p_val.min()
    p_max = p_val.max()
    step_size = (p_val.max() - p_val.min()) / 1000

    epsilon = p_min
    while epsilon < p_max:
        predictions = p_val < epsilon
        false_positives = np.sum(np.logical_and((predictions == 1), (y_val == 0)))
        true_positives = np.sum(np.logical_and((predictions == 1), (y_val == 1)))
        false_negatives = np.sum(np.logical_and((predictions == 0), (y_val == 1)))
        precision = compute_precision(true_positives, false_positives)
        recall = compute_recall(true_positives, false_negatives)
        f1 = compute_f1(precision, recall)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_recall = recall
            best_f1_precision = precision
            best_epsilon = epsilon

        epsilon = epsilon + step_size

    return best_epsilon, best_f1, best_f1_recall, best_f1_precision


def visualize_fit(X, mu, sigma2):
    n = np.arange(0, 35.5, 0.5)
    x1,x2 = np.meshgrid(n,n)

    Z = multivariate_gaussian_v1(np.column_stack((x1.reshape(5041,1), x2.reshape(5041, 1))), mu, sigma2)
    print(Z.shape)
    Z = Z.reshape(x1.shape)
    plt.plot(X[:,0], X[:,1], 'bx')

    if np.isinf(Z).sum() == 0:
        plt.contour(x1,x2,Z)


def version_one_2d(server_data):
    X = server_data.get('X')
    y_val = server_data.get('yval')
    y_val = np.reshape(y_val, y_val.shape[0])
    labels = ["Latency", "Throughput"]
    plot_distributions(server_data.get('X'), labels)
    mu, sigma_sqr = estimate_gaussian(X)
    p = multivariate_gaussian_v1(X, mu, sigma_sqr)
    visualize_fit(X, mu, sigma_sqr)

    x_val = server_data.get('Xval')
    p_val = multivariate_gaussian_v1(x_val, mu, sigma_sqr)
    (epsilon, f1, recall, precision) = select_threshold(y_val, p_val)
    print("epsilon= ", epsilon)
    print("f1= ", f1)
    print("precision= ", precision)
    print("recall= ", recall)
    outliers = np.where(p < epsilon)
    print("Outliers found:  " + repr(outliers[0].size))

    plot1 = X[outliers, 0][0]
    plot2 = X[outliers, 1][0]
    plt.plot(plot1, plot2, 'ro', mfc='none', linewidth=2, markersize=10)
    plt.show()


def version_one_multi_d(larger_data):

    # 1000 samples with each sample having 11 features
    X = larger_data.get('X')
    y_val = larger_data.get('yval')
    y_val = np.reshape(y_val, y_val.shape[0])
    mu, sigma_sqr = estimate_gaussian(X)
    # print("mu= ", mu)
    print("Sigma (sigma_sqr)", sigma_sqr)
    p = multivariate_gaussian_v1(X, mu, sigma_sqr)
    x_val = larger_data.get('Xval')
    p_val = multivariate_gaussian_v1(x_val, mu, sigma_sqr)
    (epsilon, f1, recall, precision) = select_threshold(y_val, p_val)
    print("epsilon= ", epsilon)
    print("f1= ", f1)
    print("precision= ", precision)
    print("recall= ", recall)
    outliers = np.where(p < epsilon)
    print("Outliers found:  " + repr(outliers[0].size))


def version_two_2d(server_data):

    X = server_data.get('X')
    x_transpose = np.transpose(X)
    print(x_transpose.shape)
    # N samples are now in column vectors with k feature rows
    # compute the means of the rows which hold the same features
    # mu should be a column vector of length k (k = number of features)
    mu = np.mean(x_transpose, axis=1)
    print("mu shape=" + repr(mu.shape))

    # compute the covariance matrix sigma
    sigma = np.cov(x_transpose, bias=True)
    print(sigma, sigma.shape)
    visualize_fit(X, mu, sigma)

    p = multivariate_normal.pdf(X, mu, sigma)
    # p = multivariate_normal.pdf(X, mu, sigma.diagonal())
    print(p.shape)
    x_cross_validation = server_data.get('Xval')
    y_val = server_data.get('yval')
    y_val = np.reshape(y_val, y_val.shape[0])
    p_val = multivariate_normal.pdf(x_cross_validation, mu, sigma)
    # p_val = multivariate_normal.pdf(x_cross_validation, mu, sigma.diagonal())
    (epsilon, f1, recall, precision) = select_threshold(y_val, p_val)
    print("epsilon= ", epsilon)
    print("f1= ", f1)
    print("precision= ", precision)
    print("recall= ", recall)
    outliers = np.where(p < epsilon)
    print("Outliers found:  " + repr(outliers[0].size))
    plot1 = X[outliers, 0][0]
    plot2 = X[outliers, 1][0]
    plt.plot(plot1, plot2, 'ro', mfc='none', linewidth=2, markersize=10)
    plt.show()


def version_two_multi_d(server_data):

    X = server_data.get('X')
    x_transpose = np.transpose(X)
    print(x_transpose.shape)
    # N samples are now in column vectors with k feature rows
    # compute the means of the rows which hold the same features
    # mu should be a column vector of length k (k = number of features)
    mu = np.mean(x_transpose, axis=1)
    #print("mu= ", mu)

    # compute the covariance matrix sigma
    sigma = np.cov(x_transpose, bias=True)
    # print("Sigma=", sigma, sigma.shape)
    print("Sigma diagonal", sigma.diagonal())

    # p = multivariate_normal.pdf(X, mu, sigma.diagonal())
    p = multivariate_normal.pdf(X, mu, sigma)
    # print(p)
    x_cross_validation = server_data.get('Xval')
    y_val = server_data.get('yval')
    y_val = np.reshape(y_val, y_val.shape[0])
    # p_val = multivariate_normal.pdf(x_cross_validation, mu, sigma.diagonal())
    p_val = multivariate_normal.pdf(x_cross_validation, mu, sigma)
    (epsilon, f1, recall, precision) = select_threshold(y_val, p_val)
    print("epsilon= ", epsilon)
    print("f1= ", f1)
    print("precision= ", precision)
    print("recall= ", recall)
    outliers = np.where(p < epsilon)
    print("Number of outliers found:  " + repr(outliers[0].size))


def run_driver():
    # data_small = read_matlab_data('data/server_latency_throughput.mat')
    data_large = read_matlab_data('data/larger_data_set.mat')
    # version_one_2d(data_small)
    # version_one_multi_d(data_large)
    # version_two_2d(data_small)
    version_two_multi_d(data_large)


if __name__ == "__main__":
    run_driver()

