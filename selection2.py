# To perform supervised feature selection with the Fisher criterion
# import statements:
#   sys for defining and retrieving program arguments
#   numpy to import and perform matrix operations with given data
import sys as system
import numpy as numpython


# Function: between_scatter
# Parameters: matrix = input data, groups = input labels
#   To compute the between class scatter feature wise
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
    B_value = numpython.zeros([1, avg.shape[0]])
    for count in range(group_count):
        for row in range(rows):
            B_value[0, row] += group_size[0, count] * ((group_avg[row, count] - avg[row, 0]) ** 2)

    # Compute the feature wise group wise scatter
    scatter = numpython.zeros(group_avg.shape)
    for group in unique_groups:
        for col in range(columns):
            if groups[col] == group:
                for row in range(rows):
                    scatter[row, group_index[group]] += (matrix[row, col] - group_avg[row, group_index[group]]) ** 2

    # return Between class matrix
    return B_value.flatten(), scatter


# Function: compute_rank
# Parameters: B_value = feature wise group wise between class scatter, scatter = feature wise group wise scatter
#   To compute the rank for features based on Fisher's criterion
def compute_rank(B_value, scatter):
    # Compute the overall scatter
    scatter_w = numpython.sum(scatter, axis=1)

    # Compute the rank for each features
    feature_rank = [0] * len(scatter_w)
    for each_feature in range(len(feature_rank)):
        feature_rank[each_feature] = B_value[each_feature] / scatter_w[each_feature]

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

    # Call the required function to compute the between class and scatter value for each feature and label
    between_scatter_value, scatter_value = between_scatter(data, label)

    # Call the function to compute the rank of features
    rank = compute_rank(between_scatter_value, scatter_value)

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
