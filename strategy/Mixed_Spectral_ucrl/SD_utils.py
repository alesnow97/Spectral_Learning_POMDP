import numpy as np


def generate_random_unit_vectors_in_sphere(L, D):
    # generate L random samples from the unit sphere
    random_samples = np.random.normal(size=(L, D))
    # the default norm from the function is the L2 norm
    random_unit_vectors_in_sphere = (
        random_samples / np.linalg.norm(random_samples, axis=1)[:, None]
    )

    return random_unit_vectors_in_sphere


def compute_power_iteration_update(number_of_iterations, initial_vector, tensor):

    current_vector = initial_vector
    for j in range(number_of_iterations):

        intermediate_res = np.matmul(tensor, current_vector)
        theta_bar = np.matmul(intermediate_res, current_vector)

        theta = theta_bar / np.linalg.norm(theta_bar)
        current_vector = theta

    return current_vector


def compute_eigenvalue_from_eigenvector(eigenvector, tensor):

    intermediate_res = np.matmul(tensor, eigenvector)
    intermediate_res = np.matmul(intermediate_res, eigenvector)
    eigenvalue = np.dot(intermediate_res, eigenvector)

    return eigenvalue


def compute_estimated_eigenvector_eigenvalue_pair(
    num_dimensions,  # this is the size of each generated random vector
    num_initial_vectors,
    num_iterations,
    tensor,
):

    random_unit_vectors_in_sphere = generate_random_unit_vectors_in_sphere(
        num_iterations, num_dimensions
    )

    projected_eigenvectors = np.empty(shape=(num_initial_vectors, num_dimensions))
    projected_eigenvalues = np.empty(shape=num_initial_vectors)

    for i in range(num_initial_vectors):
        current_vector = random_unit_vectors_in_sphere[i]

        iterated_vector = compute_power_iteration_update(
            number_of_iterations=num_iterations,
            initial_vector=current_vector,
            tensor=tensor,
        )

        computed_eigenvalue = compute_eigenvalue_from_eigenvector(
            eigenvector=iterated_vector, tensor=tensor
        )

        projected_eigenvalues[i] = computed_eigenvalue
        projected_eigenvectors[i] = iterated_vector

    index_of_max = np.argmax(projected_eigenvalues)
    selected_eigenvector = projected_eigenvectors[index_of_max]

    # here we should repeat the projection operation multiple times
    estimated_eigenvector = compute_power_iteration_update(
        number_of_iterations=num_iterations,
        initial_vector=selected_eigenvector,
        tensor=tensor,
    )

    estimated_eigenvalue = compute_eigenvalue_from_eigenvector(
        eigenvector=estimated_eigenvector, tensor=tensor
    )

    return estimated_eigenvector, estimated_eigenvalue


def compute_transition_matrix_permutation(matrix, permutation):

    permuted_matrix = np.zeros_like(matrix)
    matrix_dimension = matrix.shape[0]  # the matrix is assumed to be square
    for i in range(matrix_dimension):
        for j in range(matrix_dimension):
            permuted_matrix[i, j] = matrix[permutation[i], permutation[j]]

    return permuted_matrix
