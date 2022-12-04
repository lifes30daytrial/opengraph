'''
Written by:
Jonathan Xu jonathanxu29@gmail.com

Uses the to_precision library, written by:
William Rusnack github.com/BebeSparkelSparkel linkedin.com/in/williamrusnack williamrusnack@gmail.com
Eric Moyer github.com/epmoyer eric@lemoncrab.com
Thomas Hladish https://github.com/tjhladish tjhladish@utexas.edu
'''
from scipy.sparse.csgraph import _validation
from matplotlib import pyplot as plt
from matplotlib import ticker
from math import floor, log10
import to_precision as prc
import numpy as np 
import pandas as pd
import os


def linreg(x_array, y_array):
    slope = 0.000000000001
    y_int = 0
    learning_rate = 1/980
    epochs = 100000
    for i in range(epochs):
        derivative_slope = 0
        derivative_y_int = 0
        for index, value_x in np.ndenumerate(x_array):
            value_y = y_array[index]
            derivative_slope += (2 * ( ( slope * value_x ) + y_int - value_y ) * value_x )
            derivative_y_int += (2 * ( ( slope * value_x ) + y_int - value_y ) )
        slope -= learning_rate * derivative_slope / len(x)
        y_int -= learning_rate * derivative_y_int / len(x)
    return slope, y_int


@ticker.FuncFormatter
def major_formatter(x, pos):
    return prc.std_notation(x, sig_digs)


data = pd.read_csv(".\\graph.csv", header=None)
graph_title =  data[0][4]
x_axis = data[0][7]
y_axis = data[0][10]
sig_digs = int(data[8][12])

x_raw = pd.Series.to_numpy(data[0][13::])
y_raw = pd.Series.to_numpy(data[1][13::])
x = x_raw.astype(np.float32)
y = y_raw.astype(np.float32)
x = x[np.logical_not(np.isnan(x))]
y = y[np.logical_not(np.isnan(y))]

x_uncert_raw = pd.Series.to_numpy(data[3][13::])
y_uncert_raw = pd.Series.to_numpy(data[4][13::])
x_uncert = x_uncert_raw.astype(np.float32)
y_uncert = y_uncert_raw.astype(np.float32)
x_uncert = x_uncert[np.logical_not(np.isnan(x_uncert))]
y_uncert = y_uncert[np.logical_not(np.isnan(y_uncert))]

graph_x_range = float(data[7][5]), float(data[7][6])
graph_y_range = float(data[7][9]), float(data[7][10])

max_x1 = float(data[8][18])
max_y1 = float(data[8][19])
max_x2 = float(data[8][20])
max_y2 = float(data[8][21])

min_x1 = float(data[12][18])
min_y1 = float(data[12][19])
min_x2 = float(data[12][20])
min_y2 = float(data[12][21])

fig, ax = plt.subplots()

ax.errorbar(x, y, 
            xerr = x_uncert, yerr = y_uncert,
            fmt = '.k', capsize = 1, elinewidth = 1)

ax.xaxis.set_major_formatter(major_formatter)
ax.yaxis.set_major_formatter(major_formatter)

ax.set_xlim(graph_x_range[0], graph_x_range[1])
ax.set_ylim(graph_y_range[0], graph_y_range[1])

ax.set_title(graph_title)
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)

#line of best fit
slope, yint = linreg(x, y)
ax.axline((0, yint), slope=slope, label="Line of Best Fit")


low_x = np.min(x)
low_y = y[np.where(x == low_x)][0]

high_x = np.max(x)
high_y = y[np.where(x == high_x)][0]
#min line
ax.axline((low_x + min_x1, low_y + min_y1), (high_x + min_x2, high_y + min_y2), color='orange', label="Min Line")

#max line
ax.axline((low_x + max_x1, low_y + max_y1), (high_x + max_x2, high_y + max_y2), color='grey', label="Max Line")

ax.legend()

plt.grid()

lobf_slope = slope
lobf_y_int = yint
min_slope = ( ( high_y + min_y2 ) - ( low_y + min_y1 ) ) / ( ( high_x + min_x2 ) - ( low_x + min_x1 ) )
min_y_int = ( ( high_y + min_y2 ) - min_slope * ( high_x + min_x2 ) )

max_slope = ( ( high_y + max_y2 ) - ( low_y + max_y1 ) ) / ( ( high_x + max_x2 ) - ( low_x + max_x1 ) )
max_y_int = ( ( high_y + max_y2 ) - max_slope * ( high_x + max_x2 ) )

#saving

path = ".\\opengraph"

try:
    os.mkdir(path)
except OSError as error:
    pass

savefile = open(".\\opengraph\\graph_info.txt", "w")
lines = [f"Line of Best Fit - Slope: {lobf_slope} | Y-Intercept: {lobf_y_int}\n",
        f"Min Line - Slope {min_slope} | Y-Intercept: {min_y_int}\n",
        f"Max Line - Slope {max_slope} | Y-Intercept: {max_y_int}"]
savefile.writelines(lines)
savefile.close

plt.savefig(".\\opengraph\\graph.png")

