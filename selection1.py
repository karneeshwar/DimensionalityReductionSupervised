# To perform supervised feature selection with the Pearson correlation criterion
# import statements:
#   sys for defining and retrieving program arguments
#   numpy to import and perform matrix operations with given data
import sys as system
import numpy as numpython


# Function: scatter_data
# Parameters: matrix = input data
#   To compute the scatter of the given input data
def scatter_data(matrix):
    # Compute average and subtract it from the input matrix
    avg = numpython.mean(matrix, axis=1)
    matrix_avg = matrix - avg

    # Compute the scatter
    rows, columns = matrix_avg.shape
    scatter = [0] * rows
    for row in range(rows):
        for column in range(columns):
            scatter[row] += matrix_avg[row, column] ** 2

    # Return scatter of the data
    return scatter


# Function: scatter_label
# Parameters: group = input label
#   To compute the scatter of the given input label
def scatter_label(group):
    # Compute average and subtract it from each element
    avg = numpython.mean(group)
    group_avg = group - avg

    # Compute the scatter
    count = group_avg.shape[0]
    scatter = 0
    for each_label in range(count):
        scatter += group_avg[each_label] ** 2

    # Return scatter of the labels
    return scatter


# Function: pearson_correlation
# Parameters: matrix = input data, group = input label
#   To compute the pearson correlation coefficient of the given data
def pearson_correlation(matrix, group):
    # Compute average and subtract it from the input matrix
    avg = numpython.mean(matrix, axis=1)
    matrix_avg = matrix - avg

    # Compute average and subtract it from each element
    avg = numpython.mean(group)
    group_avg = group - avg

    # Compute the pearson correlation
    pearson_corr = numpython.array(group_avg * matrix_avg.T).flatten()

    # Return pearson correlation coefficient
    return pearson_corr


# Function: compute_rank
# Parameters: matrix = input data, group = input label
#   To compute the pearson correlation coefficient of the given data
def compute_rank(pearson_corr, scatter_d, scatter_l):
    # Compute the rank for each column
    feature_rank = [0] * len(scatter_d)
    for each_feature in range(len(feature_rank)):
        feature_rank[each_feature] = abs(pearson_corr[each_feature] / ((scatter_d[each_feature] * scatter_l) ** 0.5))

    # Return the rank
    return feature_rank


# Function: reduce_data
# Parameters: matrix = input data, r1 = rank 1 column number, r2 = rank 2 column number
#   To find and return the columns corresponding to rank 1 and rank 2
def reduce_data(matrix, r1, r2):
    # Copy the top features into the first to columns of the matrix
    matrix[:, 0], matrix[:, r1] = matrix[:, r1], matrix[:, 0]
    matrix[:, 1], matrix[:, r2] = matrix[:, r2], matrix[:, 1]

    # Return the top 2 selected features
    return matrix[:, 0:2]


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

    # Call the required function to compute the column wise scatter of the data
    data_scatter = scatter_data(data)

    # Call the required function to compute the scatter of the label
    label_scatter = scatter_label(label)

    # Call the required function to compute pearson correlation
    coefficient = pearson_correlation(data, label)

    # Call the function to compute the rank of features
    rank = compute_rank(coefficient, data_scatter, label_scatter)

    # Find the top 2 selected column numbers from rank
    col_rank = []
    for col_num in range(data.shape[0]):
        col_rank.append((rank[col_num], col_num))
    col_rank.sort(reverse=True)
    rank1 = col_rank[0][1]
    rank2 = col_rank[1][1]

    # Compute the reduced data
    reduced_data = reduce_data(data.T, rank1, rank2)

    # Exception handling for output data file
    while 1:
        try:
            numpython.savetxt(system.argv[3], reduced_data, delimiter=',')
            break
        except IOError:
            print('File not written')
