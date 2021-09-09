from statistics import mean
from matplotlib import style
import numpy as np
import matplotlib.pyplot as plt
import random

style.use("fivethirtyeight")

# x values
xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# y values
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def create_dataset(count, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(count):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == "pos":
            val += step
        elif correlation and correlation == "neg":
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / (
        (mean(xs) * mean(xs)) - mean(xs * xs)
    )
    b = mean(ys) - (m * mean(xs))
    return m, b


# difference between regression line and actual ys
def squared_error(ys_original, ys_line):
    return sum((ys_line - ys_original) ** 2)


# find r^2 value
def coefficient_of_determination(ys_original, ys_line):
    y_mean_line = [mean(ys_original) for y in ys_original]
    squared_error_regr = squared_error(ys_original, ys_line)
    squared_error_y_mean = squared_error(ys_original, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


xs, ys = create_dataset(40, 15, 2, correlation="pos")

m, b = best_fit_slope_and_intercept(xs, ys)

# create regression line of y values based on all xs
# equivalent to:
# 	for x in xs:
# 		regression_line.append((m * x) + b)
regression_line = [(m * x) + b for x in xs]

# predict y where x = 8
predict_x = len(xs) + 10
predict_y = (m * predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)

print(f"m: {m}")
print(f"b: {b}")
print(f"R^2: {r_squared}")

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color="GREEN")
plt.plot(xs, regression_line)
plt.show()
