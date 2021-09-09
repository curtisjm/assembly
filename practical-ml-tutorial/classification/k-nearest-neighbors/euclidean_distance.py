from math import sqrt

plot1 = [1, 3]
plot2 = [2, 5]

euclidean_distance = sqrt(sum([(a - b) ** 2 for a, b in zip(plot1, plot2)]))

print(euclidean_distance)
