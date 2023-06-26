import numpy as np


def newton_1d_bnd(funfcn, df, df2, ax, bx):
    deriv = np.inf
    x = 0
    x_old = np.inf

    while np.abs(deriv) > 0.001 and np.abs(x - x_old) > 1e-3:
        x_old = x
        deriv = df(x)
        if x == ax and deriv < 0:
            break
        elif x == bx and deriv > 0:
            break
        if np.abs(deriv) > 1e3:
            update = 0.01 * np.sign(deriv)
        else:
            deriv2 = df2(x)
            update = -deriv / deriv2

        x += update
        x = max(min(x, bx), ax)

    fval = funfcn(x)

    return x, fval
