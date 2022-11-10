# To perform supervised feature selection with the recursive elimination technique, applied with a linear discriminant
# import statements:
#   sys for defining and retrieving program arguments
#   numpy to import and perform matrix operations with given data
import sys as system
import numpy as numpython

# Function: main
#   checks for arguments, imports data and performs necessary operations for dimensionality reduction
if __name__ == '__main__':
    # If the number of arguments are incorrect prompt user to rerun program and exit
    if len(system.argv) != 4:
        print('Incorrect arguments, rerun the program with correct number of arguments!')
        system.exit()

    # Exception handling for input files
    while 1:
        try:
            data = numpython.asmatrix(numpython.genfromtxt(system.argv[1], delimiter=',', autostrip=True))
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

    # Introduce bias as column 0 into the input data
    bias = numpython.ones([data.shape[0], 1])
    data = numpython.hstack((bias, data))
    features = data.shape[1]

    # Compute the best least square solution and eliminate feature that has the least value
    # Repeat until there is only 2 features remaining
    while features > 3:
        least_squares = numpython.linalg.lstsq(data, label, rcond=None)[0]
        least_squares = least_squares[1:]
        minimum = float('inf')
        remove = -1
        for feature_number in range(len(least_squares)):
            if least_squares[feature_number] < minimum:
                minimum = least_squares[feature_number]
                remove = feature_number
        if remove >= 0:
            data = numpython.delete(data, remove + 1, 1)
        features -= 1

    # Compute the reduced data discarding the bias column in the data
    reduced_data = data[:, 1:]

    # Exception handling for output data file
    while 1:
        try:
            numpython.savetxt(system.argv[3], reduced_data, delimiter=',')
            break
        except IOError:
            print('File not written')
