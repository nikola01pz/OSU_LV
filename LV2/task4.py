import numpy as np
import matplotlib.pyplot as plt

bright_square1 = np.ones((50,50))
bright_square2 = bright_square1.copy()
dark_square1 = np.zeros((50,50))
dark_square2 = dark_square1.copy()

first_row = np.hstack((dark_square1, bright_square1))
second_row = np.hstack((bright_square2, dark_square1))
square = np.vstack((first_row, second_row))
plt.figure()
plt.imshow(square, cmap ="gray")
plt.show()