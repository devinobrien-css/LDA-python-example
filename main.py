import numpy as np

def mybLDA_train(Xp, Xn):
    # Compute the class specific means
    mean_Xp = np.mean(Xp, axis=1, keepdims=True)
    mean_Xn = np.mean(Xn, axis=1, keepdims=True)
    
    # Compute the between-class scattering matrix
    Sb = np.dot(mean_Xp - mean_Xn, (mean_Xp - mean_Xn).T)
    
    # Compute the within-class scattering matrix
    Cp = np.cov(Xp)
    Cn = np.cov(Xn)
    Sw = Cp + Cn
    
    # Compute the eigenvalues and eigenvectors of inv(Sw)*Sb
    M = np.dot(np.linalg.inv(Sw), Sb)
    eig_vals, eig_vecs = np.linalg.eig(M)
    
    # Find the eigenvector corresponding to the largest eigenvalue
    idx = np.argmax(eig_vals)
    w = eig_vecs[:, idx]
    
    # Normalize the eigenvector to unit length
    w = w / np.linalg.norm(w)
    
    return w

def mybLDA_classify(X, v, Xp, Xn):
    """
        unfinished - couldn't resolve dimensionality issue
    """
    # project data onto the line spanned by v
    projected_data = np.dot(X.T, v)
    
    # compute distance to each class mean
    dist_to_p = np.linalg.norm(projected_data - np.dot(Xp.T, v), axis=1)
    dist_to_n = np.linalg.norm(projected_data - np.dot(Xn.T, v), axis=1)
    
    # classify each sample based on the closest class mean
    r = np.ones(X.shape[1])  # initialize all samples to be in positive class
    for i in range(X.shape[1]):
        if dist_to_n[i] < dist_to_p[i]:
            r[i] = -1
    
    return r

def main():
    print('Homework 4 : LDA in Python')

    Xp = np.array([
        [4, 2, 2, 3, 4, 6, 3, 8 ],
        [1, 4, 3, 6, 4, 2, 2, 3],
        [0, 1, 1, 0, -1, 0, 1, 0]
    ])

    Xn = np.array([
        [9, 6, 9, 8, 10],
        [10, 8, 5, 7, 8],
        [1, 0, 0, 1, -1]
    ])
    print('------------------------------')
    print('Compute class-specific means')

    # Compute class-specific means
    mean_Xp = np.mean(Xp, axis=1, keepdims=True)
    mean_Xn = np.mean(Xn, axis=1, keepdims=True)
    mean_X = np.mean(np.concatenate([Xp, Xn], axis=1), axis=1, keepdims=True)

    print('Mean Vector of Xp')
    print(mean_Xp)

    print('Mean Vector of Xn')
    print(mean_Xn)

    print('------------------------------')
    print('Compute class-specific covariance matrices')

    # Compute class-specific covariance matrices
    cov_Xp = np.cov(Xp, rowvar=False, ddof=1)
    cov_Xn = np.cov(Xn, rowvar=False, ddof=1)
    
    print('Covariance Matrix of Xp')
    print(cov_Xp)

    print('Covariance Matrix of Xn')
    print(cov_Xn)

    print('------------------------------')
    print('Compute class-specific covariance matrices')

    # Compute between-class scattering matrix
    diff_mean = mean_Xp.T - mean_Xn
    Sb = np.dot(diff_mean.T, diff_mean)
    print(Sb)

    print('------------------------------')
    print('Compute within-class scattering matrix')

    # compute centered data matrices
    Xp_c = Xp - mean_Xp
    Xn_c = Xn - mean_Xn

    # compute the within class scattering matrix
    n_p = Xp_c.shape[1]
    n_n = Xn_c.shape[1]
    Sw_p = Xp_c @ Xp_c.T / (n_p-1)
    Sw_n = Xn_c @ Xn_c.T / (n_n-1)
    Sw = Sw_p + Sw_n
    print(Sw)
    
    print()
    print('------------------------------')
    print('Compute the LDA projection for the data')

   # compute the eigenvalues and eigenvectors of Sw^-1 * Sb
    Sw_inv = np.linalg.inv(Sw)
    eig_vals, eig_vecs = np.linalg.eig(Sw_inv @ Sb)

    # sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    projection_matrix = eig_vecs[:, :2]

    # compute the LDA projection
    w = eig_vecs[:, 0]

    # print the LDA projection
    print()
    print('------------------------------')
    print("LDA projection vector: ", w)

    # Project the data onto the LDA subspace
    Xp_lda = np.matmul(projection_matrix.T, Xp - mean_X)
    Xn_lda = np.matmul(projection_matrix.T, Xn - mean_X)

    print("LDA projection matrix:\n", projection_matrix)
    print("Transformed Xp:\n", Xp_lda)
    print("Transformed Xn:\n", Xn_lda)

    print()
    print('------------------------------')
    print('Custom LDA function result:')
    print(mybLDA_train(Xp,Xn))

    X = np.array([
        [1.3, 2.4, 6.7, 2.2, 3.4, 3.2],
        [8.1, 7.6, 2.1, 1.1, 0.5, 7.4],
        [1, 2, 3, 2, 0, 2 ]
    ])

    # print(mybLDA_classify(X, 1, Xp, Xn))

if __name__ == '__main__':
    main()