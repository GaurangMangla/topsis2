import sys
import pandas as pd
import numpy as np


def topsis(input_file, weights, impacts, output_file):

    data = pd.read_csv(input_file)

    if data.shape[1] < 3:
        raise Exception("Input file must contain at least 3 columns")

    criteria_data = data.iloc[:, 1:]

    weights = np.array(weights.split(","), dtype=float)
    impacts = impacts.split(",")

    if len(weights) != criteria_data.shape[1]:
        raise Exception("Weights count must match criteria columns")

    if len(impacts) != criteria_data.shape[1]:
        raise Exception("Impacts count must match criteria columns")

    for imp in impacts:
        if imp not in ["+", "-"]:
            raise Exception("Impacts must be + or -")

    
    norm_matrix = criteria_data / np.sqrt((criteria_data ** 2).sum())

   
    weighted_matrix = norm_matrix * weights

    
    ideal_best, ideal_worst = [], []

    for i in range(criteria_data.shape[1]):
        if impacts[i] == "+":
            ideal_best.append(weighted_matrix.iloc[:, i].max())
            ideal_worst.append(weighted_matrix.iloc[:, i].min())
        else:
            ideal_best.append(weighted_matrix.iloc[:, i].min())
            ideal_worst.append(weighted_matrix.iloc[:, i].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    
    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

    
    score = dist_worst / (dist_best + dist_worst)
    rank = score.rank(ascending=False)

    data["Topsis Score"] = score
    data["Rank"] = rank.astype(int)

    data.to_csv(output_file, index=False)

    return output_file



def main():
    if len(sys.argv) != 5:
        print("Usage: topsis <input_file> <weights> <impacts> <output_file>")
        sys.exit(1)

    _, input_file, weights, impacts, output_file = sys.argv

    topsis(input_file, weights, impacts, output_file)
    print("TOPSIS Completed Successfully!")


if _name_ == "_main_":
    main()