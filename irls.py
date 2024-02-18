import numpy as np
from tqdm import trange

def IRLS(y, X, maxiter = 100, d = 1e-5, tolerance = 1e-6):
    n,p = X.shape
    delta = np.array(np.repeat(d, n) ).reshape(1,n)
    w = np.repeat(1, n)
    W = np.diag( w )
    B = np.dot(np.linalg.inv( X.T.dot(W).dot(X) ), ( X.T.dot(W).dot(y) ) )
    for _ in trange(maxiter):
        _B = B
        _w = abs(y - X.dot(B)).T
        w = float(1)/np.maximum( delta, _w )
        W = np.diag( w[0] )
        B = np.dot(np.linalg.inv( X.T.dot(W).dot(X) ), ( X.T.dot(W).dot(y) ) )
        tol = np.sum(abs( B - _B ) )
        # print("Tolerance = %s" % tol)
        if tol < tolerance:
            return B
    print("Did not converge")
    return B

if __name__ == "__main__":
    #Test Example: Fit the following data under Least Absolute Deviations regression
    # first line = "p n" where p is the number of predictors and n number of observations
    # following lines are the data lines for predictor x and response variable y
    #	 "<pred_1> ... <pred_p> y"
    # next line win "n" gives the number n of test cases to expect
    # following lines are the test cases with predictors and expected response

    input_str = '''2 7
    0.18 0.89 109.85
    1.0 0.26 155.72
    0.92 0.11 137.66
    0.07 0.37 76.17
    0.85 0.16 139.75
    0.99 0.41 162.6
    0.87 0.47 151.77
    4
    0.49 0.18 105.22
    0.57 0.83 142.68
    0.56 0.64 132.94
    0.76 0.18 129.71
    '''

    input_list = input_str.split('\n')

    p,n = [ int(i) for i in input_list.pop(0).split() ]
    X = np.empty( [n, p+1] )
    X[:,0] = np.repeat( 1, n)
    y = np.empty( [n, 1] )
    for i in range(n):
        l = [ float(i) for i in input_list.pop(0).split() ]
        X[i, 1:] = np.array( l[0:p] )
        y[i] = np.array( l[p] )

    n = [ int(i) for i in input_list.pop(0).split() ][0]
    X_new = np.empty( [n, p+1] )
    X_new[:,0] = np.repeat( 1, n)
    y_new = np.empty( [n, 1] )
    for i in range(n):
        l = [ float(i) for i in input_list.pop(0).split() ]
        X_new[i, 1:] = np.array( l[0:p] )
        y_new[i] = np.array( l[p] )


    B = IRLS(y=y,X=X, maxiter=20)
    abs_error = abs( y_new - X_new.dot(B) )

