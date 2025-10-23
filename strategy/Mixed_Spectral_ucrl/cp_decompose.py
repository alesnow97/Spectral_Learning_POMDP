"""Compute CP decomposition for 3-D tensors
"""

import numpy as np
from numpy import linalg as LA
from sklearn.utils import check_random_state


def _check_1d_vector(vector):
    """Check 1D vector shape

    Check 1D vector shape. array with shape
    [n, 1] or [n, ] are accepted. Will return
    a 1 dimension vector.

    Parameters
    ----------
    vector : array (n,) or (n, 1)
        rank one vector

    Returns
    -------
    vector : array, (n,)
    """

    v_shape = vector.shape
    if len(v_shape) == 1:
        return vector
    elif len(v_shape) == 2 and v_shape[1] == 1:
        return vector.reshape(
            v_shape[0],
        )
    else:
        raise ValueError("Vector is not 1-d array: shape %s" % str(v_shape))


def _check_square_matrix(matrix):
    """Check 2D matrix shape

    Check 1D vector shape. array with shape
    [n, 1] or [n, ] are accepted. Will return
    a 1 dimension vector.

    Parameters
    ----------
    matrix : (n, n)
        rank one vector

    Returns
    -------
    matrix : array, (n, n)
    """
    m_shape = matrix.shape

    if len(m_shape) == 2:
        if m_shape[0] != m_shape[1]:
            raise ValueError("matrix is not square: shape %s" % str(m_shape))
        return matrix
    else:
        raise ValueError("matrix is not 2-d array: shape %s" % str(m_shape))


def rank_1_tensor_3d(a, b, c):
    """Generate a 3-D tensor from 3 1-D vectors

    Generate a 3D tensor from 3 rank one vectors
    `a`, `b`, and `c`. The returned 3-D tensor is
    in unfolded format.

    Parameters
    ----------
    a : array, shape (n,)
        first rank one vector

    b : array, shape (n,)
        second rank one vector

    c : array, shape (n,)
        thrid rank one vector

    Returns
    -------
    tensor:  array, (n, n * n)
        3D tensor in unfolded format. element
        (i, j, k) will map to (i, (n * k) + j)
    """

    a = _check_1d_vector(a)
    b = _check_1d_vector(b)
    c = _check_1d_vector(c)

    dim = a.shape[0]
    # check dimension
    if (dim != b.shape[0]) or (dim != c.shape[0]):
        raise ValueError(
            "Vector dimension mismatch: (%d, %d, %d)" % (dim, b.shape[0], c.shape[0])
        )

    outter = b[:, np.newaxis] * c[:, np.newaxis].T
    tensor = a[:, np.newaxis] * outter.ravel(order="F")[np.newaxis, :]
    return tensor


def tensor_3d_from_vector_matrix(a, b):
    """Generate 3-D tensor from 1-D vector and 2-D matrix

    Generate a 3D tensor from a 1-D vector `a` and 2-D
    matrix `b`. The returned 3-D tensor is
    in unfolded format.

    Parameters
    ----------
    a : array, shape (m,)
        1-D vector

    b : 2-D array, shape (n, p)
        2-D matrix

    Returns
    -------
    tensor:  array, (m, n * p)
        3D tensor in unfolded format.

    """
    a = _check_1d_vector(a)
    tensor = a[:, np.newaxis] * b.ravel(order="F")[np.newaxis, :]
    return tensor


def tensor_3d_from_matrix_vector(b, a):
    """Generate 3-D tensor from 2-D matrix and 1-D vector

    This function is similar to `tensor_3d_from_vector_matrix`
    function. The only difference is the first argument is 2-D
    matrix and the second element is 1-D vector.

    Parameters
    ----------
    b : array, shape (m, n)
        2-D matrix

    a : array, shape (p,)
        vector

    Returns
    -------
    tensor : array, shape (m, n * p)
        3D tensor in unfolded format.

    """
    len_a = a.shape[0]
    n_col = b.shape[1]
    tensor = np.tile(b, len_a)
    for i in range(len_a):
        col_from = n_col * i
        col_to = n_col * (i + 1)
        tensor[:, col_from:col_to] *= a[i]
    return tensor


def tensor_3d_permute(tensor, tensor_shape, a, b, c):
    """Permute the mode of a 3-D tensor

    This is a slow implementation to generate 3-D tensor
    permutations.

    Parameters
    ----------
    tensor : 2D array, shape (n, m * k)
        3D tensor in unfolded format

    tensor_shape : int triple
        Shape of the tensor. Since tensor is in
        unfolded format. We need it's real format
        to calculate permutation.

    a : int, {1, 2, 3}
        new first index
    }

    b : int, {1, 2, 3}
        new second index

    c : int, {1, 2, 3}
        new thrid order index

    Return
    ------
    permuted_tensor:  2D array
        Permuted tensor, element (i_1, i_2, i_3) in
        the permuted tensor will be element
        (i_a, i_b, i_c) in the original tensor
    """

    a_idx = a - 1
    b_idx = b - 1
    c_idx = c - 1

    n_col = tensor_shape[1]
    dim1 = tensor_shape[a_idx]
    dim2 = tensor_shape[b_idx]
    dim3 = tensor_shape[c_idx]

    permuted_tensor = np.empty((dim1, dim2 * dim3))
    old_idx = np.zeros(3).astype("int32")
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                old_idx[a_idx] = i
                old_idx[b_idx] = j
                old_idx[c_idx] = k
                old_val = tensor[old_idx[0], (n_col * old_idx[2]) + old_idx[1]]
                # new index
                permuted_tensor[i, (dim2 * k) + j] = old_val

    return permuted_tensor


def khatri_rao_prod(a, b):
    """Khatri-Rao product

    Generate Khatri-Rao product from 2 2-D matrix.

    Parameters
    ----------
    a : 2D array, shape (n, k)
        first matrix

    b : 2D array, shape (m, k)
        second matrix

    Returns
    -------
    matrix : 2D array, shape (n * m, k)
        Khatri-Rao product of `a` and `b`
    """

    a_row, a_col = a.shape
    b_row, b_col = b.shape
    # check column size
    if a_col != b_col:
        raise ValueError("column dimension mismatch: %d != %d" % a_col, b_col)
    matrix = np.empty((a_row * b_row, a_col))
    for i in range(a_col):
        matrix[:, i] = np.kron(a[:, i], b[:, i])
    return matrix


def tensor_3d_prod(tensor, a, b, c):
    """Calculate product of 3D tensor with matrix on each dimension

    Parameters
    ----------
    tensor : 3D array, shape (n1, n2, n3)
    a : array, (n1, m)

    b :  array, (n2, n)

    c :  array, (n3, p)

    Returns
    -------
    t_abc : array, (m, n, p)
        tensor(a, b, c)

    """
    n1, n2, n3 = tensor.shape
    n1_, m = a.shape
    n2_, n = b.shape
    n3_, p = c.shape

    # (n1, n2, p)
    t_c = np.dot(tensor, c)

    t_bc = np.empty((n1, n, p))
    for i in range(n1):
        # (n, p) = (n, n2) * (n2, p)
        t_bc[i, :, :] = np.dot(b.T, t_c[i, :, :])

    t_abc = np.empty((m, n, p))
    for i in range(p):
        t_abc[:, :, i] = np.dot(a.T, t_bc[:, :, i])
    return t_abc


def _check_3d_tensor(tensor, n_dim):

    t_shape = tensor.shape
    n_col = n_dim * n_dim
    if len(t_shape) != 2:
        raise ValueError("dimension mismatch: tensor need be a 2-D array")

    if t_shape[0] != n_dim:
        raise ValueError("row dimension mismatch: %d != %d" % (t_shape[0], n_dim))

    if t_shape[1] != n_col:
        raise ValueError("column dimension mismatch: %d != %d" % (t_shape[1], n_col))


def _als_iteration(tensor, b, c):
    """One ALS iteration"""

    temp1 = np.dot(tensor, khatri_rao_prod(c, b))
    temp2 = LA.pinv(np.dot(c.T, c) * np.dot(b.T, b))
    a_update = np.dot(temp1, temp2)

    lambdas = LA.norm(a_update, axis=0)
    a_update /= lambdas
    return lambdas, a_update


def tensor_reconstruct(lambdas, a, b, c):
    t = np.dot(np.dot(a, np.diag(lambdas)), khatri_rao_prod(c, b).T)
    return t


def cp_als(tensor, n_component, n_restart, n_iter, tol, random_state):
    """CP Decomposition with ALS

    CP decomposition with Alternating least square (ALS) method.
    The method assume the tensor can be composed by sum of
    `n_component` rank one tensors.

    Parameters
    ----------
    tensor : array, (k, k * k)
        Symmetric Tensor to be decomposed with unfolded format.

    n_component: int
        Number of components

    n_restart: int
        Number of ALS restarts

    n_iter: int
        Number of iterations for ALS

    random_state: int, RandomState instance or None, optional, default = None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    tol : float, optional
        Tolerance

    Returns
    -------
    lambda: array, (k,)

    a: array, (k, k)

    b: array, (k, k)

    c: array, (k, k)
    """

    # check tensor shape
    _check_3d_tensor(tensor, n_component)

    converge_threshold = 1.0 - tol
    random_state = check_random_state(random_state)

    best_a = None
    best_b = None
    best_c = None
    best_lambdas = None
    best_loss = None

    for i in range(n_restart):

        # use QR factorization to get random starts
        a, _ = LA.qr(random_state.randn(n_component, n_component))
        b, _ = LA.qr(random_state.randn(n_component, n_component))
        c, _ = LA.qr(random_state.randn(n_component, n_component))

        for iteration in range(n_iter):

            # check convergence
            if iteration > 0:
                diag = np.diag(np.dot(prev_a.T, a))
                # print(diag)
                if np.all(diag > converge_threshold):
                    print("ALS converge in %d iterations" % iteration)
                    break

            prev_a = a
            _, a = _als_iteration(tensor, b, c)
            _, b = _als_iteration(tensor, c, a)
            lambdas, c = _als_iteration(tensor, a, b)

        # update optimal values
        reconstructed = tensor_reconstruct(lambdas, a, b, c)
        loss = LA.norm(tensor - reconstructed)
        print("restart: %d, loss: %.5f" % (i, loss))
        if best_loss is None or loss < best_loss:
            best_a = a
            best_b = b
            best_c = c
            best_lambdas = lambdas
            best_loss = loss
            print("ALS best loss: %.5f" % best_loss)
            if best_loss < tol:
                break

    return (best_lambdas, best_a, best_b, best_c)


def cp_tensor_power_method(tensor, n_component, n_restart, max_iter, random_state):
    """CP Decomposition with Robust Tensor Power Method

    Parameters
    ----------
    tensor : array, (k, k * k)
        Symmetric Tensor to be decomposed with unfolded format.

    n_component : int
        Number of components

    n_restart : int
        Number of ALS restarts

    max_iter : int
        Number of iterations for tensor power method

    random_state: int, RandomState instance or None, optional, default = None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    lambdas : array, (k,)
        eigen values for the tensor

    vectors : array, (k, k)
        eigen vectors for the tensor. (`vectors[:,i]` map to `lambdas[i]`)

    total_iters : int
        Total number of iterations

    """
    _check_3d_tensor(tensor, n_component)
    random_state = check_random_state(random_state)

    tensor_c = tensor.copy()
    lambdas = np.empty(n_component)
    vectors = np.empty((n_component, n_component))

    for t in range(n_component):
        best_lambda = 0.0
        best_v = None
        total_iters = 0
        for _ in range(n_restart):
            v = random_state.randn(n_component)
            v /= LA.norm(v)
            for _ in range(max_iter):
                # compute T(I,v,v) and normalize it
                v_outer = (v * v[:, np.newaxis]).ravel()
                v = (tensor_c * v_outer).sum(axis=1)
                v /= LA.norm(v)
                total_iters += 1
            # compute T(v,v,v)
            v_outer = (v * v[:, np.newaxis]).ravel()
            new_lambda = np.dot(v, (tensor_c * v_outer).sum(axis=1))

            if best_v is None or new_lambda > best_lambda:
                best_lambda = new_lambda
                best_v = v
        lambdas[t] = best_lambda
        vectors[:, t] = best_v

        # deflat
        tensor_c -= best_lambda * rank_1_tensor_3d(best_v, best_v, best_v)
    return lambdas, vectors, total_iters
