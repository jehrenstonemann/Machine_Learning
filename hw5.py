import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Q1 & Q2 & Q3
    dataframe = pd.read_csv(sys.argv[1])
    plt.plot(dataframe['year'], dataframe['days'])
    plt.xlabel('Year')
    plt.ylabel('Number of frozen days')
    plt.savefig("plot.jpg")
    year = dataframe['year'].to_list()
    day = dataframe['days'].to_list()
    X = np.ones((len(year), 2), dtype='int64')
    Y = []
    for i in range(len(year)):
        X[i][1] = year[i]
        Y.append(day[i])
    Y = np.array(Y)
    # Q3a
    print("Q3a:")
    print(X)
    # Q3b
    print("Q3b:")
    print(Y)
    # Q3c
    Z = np.dot(X.T, X)
    print("Q3c:")
    print(Z)
    # Q3d
    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)
    # Q3e
    PI = np.dot(I, X.T)
    print("Q3e:")
    print(PI)
    # Q3f
    print("Q3f:")
    hat_beta = np.dot(PI, Y)
    print(hat_beta)
    # Q4
    y_test = hat_beta[0] + np.dot(hat_beta[1], 2021)
    print("Q4: " + str(y_test))
    # Q5a
    if hat_beta[1] > 0:
        print("Q5a: >")
    elif hat_beta[1] < 0:
        print("Q5a: <")
    else:
        print("Q5a: =")
    # Q5b
    print("Q5b: < means the number of Lake Mendota ice days decreases as the year increases.")
    print("> means the number of Lake Mendota ice days increases as the year increases.")
    print("= means the number of Lake Mendota ice days remains the same as the year increases.")
    # Q6a
    x_star = (0 - hat_beta[0]) / hat_beta[1]
    print("Q6a: " + str(x_star))
    # Q6b
    print("Q6b: Compelling because the regression slope is negative, meaning there is a decrease trend in Lake Mendota "
          "ice "
          "days. "
          "Therefore, an estimate of " + str(x_star) +
          " years tends to be a reasonable prediction of when there will be no ice "
          "left")
