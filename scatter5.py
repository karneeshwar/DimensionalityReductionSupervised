# To minimize the between-class scatter
# import statements:
#   sys for defining and retrieving program arguments
#   numpy to import and perform matrix operations with given data
import sys as system
import numpy as numpython


# Function: between_scatter
# Parameters: matrix = input data, groups = input labels
#   To compute the between class scatter matrix
def between_scatter(matrix, groups):
    # Compute average
    avg = numpython.mean(matrix, axis=1)

    # Compute group average based on labels
    unique_groups = numpython.unique(groups)
    group_count = unique_groups.shape[0]
    rows, columns = matrix.shape
    group_sum = numpython.zeros([rows, group_count])
    group_size = numpython.zeros([1, group_count])
    group_index = {}
    group_counter = 0

    for group in unique_groups:
        group_index[group] = group_counter
        group_counter += 1
        for col in range(columns):
            if groups[col] == group:
                for row in range(rows):
                    group_sum[row, group_index[group]] += matrix[row, col]
                group_size[0, group_index[group]] += 1

    group_avg = numpython.asmatrix(group_sum/group_size)

    # Compute the Between class matrix
    size = avg.shape[0]
    B_matrix = numpython.zeros([size, size])
    for count in range(group_count):
        B_matrix += group_size[0, count] * (group_avg[:, count] - avg) * (group_avg[:, count] - avg).T

    # return Between class matrix
    return B_matrix


# Function: minimize_scatter
# Parameters: matrix = input data, reduce_to = number of dimensions to reduce to (default = 2)
#   To find the top 2 eigen vectors that minimizes the given scatter
def minimize_scatter(matrix, reduce_to=2):
    # Compute eigen values and vectors of the symmetric matrix B
    eigen_values, eigen_vectors = numpython.linalg.eigh(matrix)

    # sort the eigen values and vectors in increasing order of the eigen values
    pivot = numpython.argsort(eigen_values)
    eigen_values = eigen_values[pivot]
    eigen_vectors = eigen_vectors[:, pivot]

    # Find the top 2 eigen vectors
    reduced_eigen_vectors = eigen_vectors[:, :reduce_to]

    # return the reduced vectors
    return reduced_eigen_vectors


# Function: reduce_data
# Parameters: v = reduced vectors, matrix = input testing data
#   To reduce the input data using the top 2 selected vectors from the eigen value decomposition
def reduce_data(v, matrix):
    # return the reduced data
    return v.T * matrix


# Function: main
#   checks for arguments, imports data and calls necessary functions for dimensionality reduction
if __name__ == '__main__':
    # If the number of arguments are incorrect prompt user to rerun program and exit
    if len(system.argv) != 4:
        print('Incorrect arguments, rerun the program with correct number of arguments!')
        system.exit()

    # Exception handling for input files
    while 1:
        try:
            data = (numpython.asmatrix(numpython.genfromtxt(system.argv[1], delimiter=',', autostrip=True))).T
            break
        except IOError:
            print('Input data file not found')
            system.exit()
    while 1:
        try:
            label = numpython.genfromtxt(system.argv[2], delimiter=',', autostrip=True, dtype='int')
            break
        except IOError:
            print('Input labels file not found')
            system.exit()

    # Call the required function to compute the between matrix
    between_scatter_matrix = between_scatter(data, label)

    # Call the required function to compute eigen vectors that minimizes the mixture scatter
    vectors = minimize_scatter(between_scatter_matrix)

    # Call the function to compute the final reduced data
    reduced_data = reduce_data(vectors, data)

    # Exception handling for output data file
    while 1:
        try:
            numpython.savetxt(system.argv[3], reduced_data.T, delimiter=',')
            break
        except IOError:
            print('File not written')
