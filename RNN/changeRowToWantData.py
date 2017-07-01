import csv
import numpy as np


def replaceList(filename):

    list = np.loadtxt(filename, dtype=str)
    result = []
    for i in range(len(list)):
        result.append(list[i].replace("\"", ""))
    # print(result)
    for i in range(len(result)):
        print(result[i])

    return result

def writeCsvFile(filename, result):
    f = open(filename, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)

    for j in range(len(result)):
        wr.writerow(result[j])
    f.close()


result = replaceList("data.csv")
writeCsvFile("output2.csv", result)