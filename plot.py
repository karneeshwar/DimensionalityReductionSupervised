#   If you see the error: ImportError: No module named 'matplotlib'
#   then you need to install matplotlib by doing: 
#   python -mpip install -U matplotlib

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import os

if len(sys.argv) != 3:
    print('Incorrect arguments, rerun the program with correct number of arguments!')
    sys.exit()

X = np.genfromtxt(sys.argv[1], delimiter=',', autostrip=True)
assert (X.shape[1] >= 2)  # X should be at least 2D. Take its first 2 columns
x1 = X[:, 0]  # first column of X
x2 = X[:, 1]  # second column of X

y = np.genfromtxt(sys.argv[2], dtype='int')

fig = plt.figure()
col = ListedColormap(['r', 'g', 'b'])
sca = plt.scatter(x1, x2, c=list(y), cmap=col)
plt.legend(handles=sca.legend_elements()[0], labels=list(np.unique(y)))
plt.title(os.path.splitext(sys.argv[1])[0])

# fig, ax = plt.subplots()
# ax.scatter(x1, x2)
#
# if len(sys.argv) == 3:
#     for i, txt in enumerate(y):
#         ax.annotate(txt, (x1[i], x2[i]))

# save to pdf
plotname = os.path.splitext(sys.argv[1])[0] + '_plot'
pdf = PdfPages(plotname + '.pdf')
pdf.savefig(fig)
pdf.close()
