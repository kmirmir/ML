import csv



def getInfoFromCsv(csvName):
    list = []

    with open(csvName, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            list.append(row)

    return list


list = getInfoFromCsv("data.csv")
# print(list)
result = []
# 0일에서 24시간이 1일이니까 x일 동안
for i in range(0, 24*5):
    # print(list[i])
    result.append(list[i])

print("=======")
print("=======")
print("=======")
print(result[:])
print(result[:,[-1]])

print("=======")
print("=======")
print("=======")

dataX = []
dataY = []

# for i in range(len(result)):
    # print(result[i])