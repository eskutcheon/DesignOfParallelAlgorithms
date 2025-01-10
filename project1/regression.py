import numpy as np
import matplotlib.pyplot as plt
import sys


class LSRegression:
    def __init__(self, x,y):
        self.xdata = x
        self.ydata = y
    # gonna need to modify later to:
    # make less ill-defined, allow for forecasting, add new points, and test for degree of best fit

    def polyEval(self, xx): # xx contains arguments in which to evaluate the polynomial
        # returns y-values at these points in xx
        '''if !(hasattr(self, 'coef')):
            self.coef = self.polyRegression()'''
        yy = np.zeros(len(xx)) # to store approximations
        for i in range(len(self.coef)):
            yy += self.coef[i]*(xx**i) # adds Ci*x**i pairs to yy one term (of the polynomial) at a time
        return yy

    def polyRegression(self, deg): # in least squares sense
        # solves the system of normal equations V'Va = V'y where V is the Vandermonde matrix of x, V' is its transpose, and a contains coefficients
            # of the approximated polynomial - often simplified to Sa = b where S = V'V and b = V'y
            # reference found here: https://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html
        # note: works fine for linear regression when deg = 1
        if len(self.xdata) != len(self.ydata):
            print("arrays of input and output data must be the same length!")
            sys.exit(1)
        size = len(self.xdata)
        s = np.zeros(2*deg+1) # representing row elements of S = V'V
        b = np.zeros(deg+1)     # elements of b = V'y
        Smatrix = np.zeros((deg+1,deg+1)) # preallocating S
        for k in range(len(s)):
            for i in range(size):
                if k <= deg:
                    b[k] += (self.xdata[i]**k)*(self.ydata[i])
                s[k] += self.xdata[i]**k
        # filling S like so is easier than using the Vandermonde matrix and its transpose
        for i in range(deg+1):
            Smatrix[i] = s[i:(deg+i+1)] # increments along rows and columns so that skew diagonals of S are filled with s[1]
        self.coef = np.linalg.solve(Smatrix, b) # solving for the a values in the augmented matrix Sa = b
        # coefficients stored in lowest degree order first, i.e., [c0,c1,c2,...] for c0 + c1*x + c2*x**2 + ...

def print_eq(Tp, T1, p):
    const_term = Tp + T1/p
    coef = T1*(1 - 1/p)
    print(f'Tc = {const_term} + {coef}Fs')
    return coef, const_term




results = [38.275, 25.987, 16.861, 13.691, 10.979, 11.789, 3.0921]
processes = [1, 2, 4, 8, 16, 20, 40]

print('Tc = 0')
length = len(results)
new_x = np.zeros(length-1)
new_y = np.zeros(length-1)
for i in range(1, length):
    new_x[i-1], new_y[i-1] = print_eq(results[i], results[0], processes[i])
    #new_x[i-1] *= -1

reg = LSRegression(new_x, new_y)
reg.polyRegression(1)
xx = np.linspace(processes[1], processes[-1], 1000)
yy = reg.polyEval(xx)

plt.plot(xx, yy)
plt.show()
