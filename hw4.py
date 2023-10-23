import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from scipy.cluster import hierarchy


# done
def load_data(filepath):
    result = []
    with open(filepath, encoding="utf-8") as csvfile:
        content = csv.DictReader(csvfile)
        for row in content:
            dic = dict()
            dic["#"] = row['#']
            dic["Name"] = row['Name']
            dic["Type 1"] = row['Type 1']
            dic["Type 2"] = row['Type 2']
            dic["Total"] = row['Total']
            dic["HP"] = row['HP']
            dic["Attack"] = row['Attack']
            dic["Defense"] = row['Defense']
            dic["Sp. Atk"] = row['Sp. Atk']
            dic["Sp. Def"] = row['Sp. Def']
            dic["Speed"] = row['Speed']
            dic["Generation"] = row['Generation']
            dic["Legendary"] = row['Legendary']
            result.append(dic)
    return result


# done
def calc_features(row):
    result = np.array(np.zeros(6), dtype=np.int64)
    result[0] = int(row['Attack'])
    result[1] = int(row['Sp. Atk'])
    result[2] = int(row['Speed'])
    result[3] = int(row['Defense'])
    result[4] = int(row['Sp. Def'])
    result[5] = int(row['HP'])
    return result


# done
def hac(features):
    distance_matrix = np.empty([len(features), len(features)])
    individual_data = dict()
    for label in range(len(features)):
        individual_data[label] = [label]
    for i in range(len(features)):
        for j in range(len(features)):
            if i != j:
                distance_matrix[i][j] = np.linalg.norm(features[i] - features[j])
    Z = np.empty([0, 4])
    index = len(features)
    for n in range(len(features) - 1):
        minimum = float('inf')
        for i in individual_data:
            for j in individual_data:
                if i != j:
                    cluster_i = individual_data[i]
                    cluster_j = individual_data[j]
                    maximum = float('-inf')
                    for c_i in cluster_i:
                        for c_j in cluster_j:
                            maximum = max(maximum, distance_matrix[c_i][c_j])
                    if minimum > maximum or (minimum == maximum and z0 > i):
                        z0 = i
                        z1 = j
                    elif z0 == i and z1 > j:
                        z1 = j
                    minimum = min(maximum, minimum)
        individual_data[index] = individual_data[z0] + individual_data[z1]
        del individual_data[z0]
        del individual_data[z1]
        Z = np.vstack([Z, [z0, z1, minimum, len(individual_data[index])]])
        index += 1
    return Z


# done
def imshow_hac(Z):
    plt.figure()
    hierarchy.dendrogram(Z)
    plt.show()
