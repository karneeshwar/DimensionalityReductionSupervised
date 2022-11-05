# To minimize the mixture-class scatter
# import statements:
#   sys for defining and retrieving program arguments
#   numpy to import and perform matrix operations with given data
import sys as system
import numpy as numpython


# Function: mixture_scatter
# Parameters: matrix = input data
#   To compute the mixture scatter matrix
def mixture_scatter(matrix):
    # Compute average and subtract it from the input matrix
    avg = numpython.mean(matrix, axis=1)
    matrix_avg = matrix - avg

    # Compute the Mixture matrix
    M_matrix = matrix_avg * matrix_avg.T

    # Return Mixture matrix
    return M_matrix


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
            label = numpython.asmatrix(numpython.genfromtxt(system.argv[2], delimiter=',', autostrip=True))
            break
        except IOError:
            print('Input labels file not found')
            system.exit()

    # Call the required function to compute the mixture matrix
    mixture_scatter_matrix = mixture_scatter(data)

    # Call the required function to compute eigen vectors that minimizes the mixture scatter
    vectors = minimize_scatter(mixture_scatter_matrix)

    # Call the function to compute the final reduced data
    reduced_data = reduce_data(vectors, data)

    # Exception handling for output data file
    while 1:
        try:
            numpython.savetxt(system.argv[3], reduced_data.T, delimiter=',')
            break
        except IOError:
            print('File not written')
