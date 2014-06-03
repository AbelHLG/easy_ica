"""
Independent component analysis (ICA) using maximum likelihood, square mixing matrix and no noise (Infomax).
Source prior is assumed to be p(s)=1/pi*exp(-ln(cosh(s))). For optimization the BFGS algorithm is used.

Reference:
A. Bell and T.J. Sejnowski(1995).
An Information-Maximization Approach to Blind Separation and Blind Deconvolution
Neural Computation, 7:1129-1159.

History:

- 2002.4.1 created for Matlab by Thomas Kolenda of IMM, Technical University of Denmark.
- 2014.6.3 ported to Python by Abel Antonio Fernandez Higuera
                               Roberto Antonio Becerra Garcia
                               Rodolfo Valentin Garcia Bermudez
                               (Biomedical Data Processing Group (GPDB), University of Holguin)
"""

__authors__ = [
    'Abel Antonio Fernandez Higuera <afernandezh@facinf.uho.edu.cu>',
    'Roberto Antonio Becerra Garcia <idertator@facinf.uho.edu.cu>',
    'Rodolfo Valentin Garcia Bermudez <rodolfo@facinf.uho.edu.cu>',
]

from math import pi, e, sqrt

from numpy import max, min, eye, diag, var, sort, linalg, diff, float, inf, dot, isreal, any, \
    nonzero, ones, zeros, shape, array
import numpy as np


#######################################################################################
#                                AUXILIARY FUNCTIONS                                  #
#######################################################################################


def check(x0, opts0, w=None, x=None, m=None, n=None):
    """
    Check function call.
    """
    x1 = []
    for i in x0:
        for j in i:
            x1.append([j])
    x1 = array(x1)
    sx = shape(x1)
    n1 = max(sx)

    if min(sx) > 1:
        print('Error: x0 should be a vector')

    f, g = ica_mlf(w, x=x, m=m, n=n)

    sf = f.shape
    sg = g.shape

    opts = []

    if any(sf) - 1 or ~isreal(f):
        print('Error: f  should be a real valued scalar.')

    if (min(sg) != 1) or (max(sg) != n1):
        print('Error: g  should be a vector of the same length as  x')

    opts0.reshape((4, 1))
    so = opts0.shape

    if (min(so) != 1) or (max(so) < 4) or any(~isreal(opts0[0:3])):
        print('Error: opts  should be a real valued vector of length 4')

    opts = opts0[0:4]
    opts = (opts[:]).T

    i = nonzero(any(opts) <= 0)

    if len(i[0]):
        len(i)
        d = [0, 1 * (e ** (-4)) * linalg.norm(g, inf), 1 * (e ** (-8)), 100]
        opts[i] = d[i]
    return x1, n1, f, g, opts


def call_svd(x, k, draw):
    """
    Reduce dimension with SVD.
    """
    global d, u, v
    m = x.shape[0]
    n = x.shape[1]

    if n > m:
        if draw == 1:
            print('Do Transpose SVD')
            v, d, u = linalg.svd(x.T, full_matrices=0)
    else:
        if draw == 1:
            u, d, v = linalg.svd(x, full_matrices=0)

    # Esto se hace pues la funcion de Numpy solo devuleve igual a Matlab el primer y el tercer resultado. Para que
    # devuelva la matriz que se quiere, se deve diagonalizar el vector que retorna la funcion svd() de Numpy.
    d = diag(d)
    dv = dot((d[0:k, 0:k]), (v[:, 0:k]).T)

    return u, dv


def ica_mlf(w, x=None, m=None, n=None):
    """
    Returns the negative log likelihood and its gradient w.r.t. W.
    """
    w = w.reshape((m, m))
    s = dot(w, x)

    # Negative log likelihood function

    f = -(n * np.log(abs(linalg.det(w))) - sum(sum(np.log(np.cosh(s)))) - n * m * np.log(pi))

    # Gradient w.r.t. W

    dw = -(n * linalg.inv(w.T) - dot(np.tanh(s), x.T))

    # Esto es el equivalente a lo que en Matlab seria dw = dw[:] que no es mas que poner en un vector todos los
    # elementos de la matriz

    dw1 = []
    for i in dw:
        for j in i:
            dw1.append([j])
    dw = array(dw1)
    f = array([f])
    return f, dw


def interpolate(xfd, n):
    """
    Minimizer of parabola given by xfd(1:2,1:3) = [a fi(a) fi'(a); b fi(b) dummy].
    """
    xfd = array(xfd)
    a = xfd[0][0]
    b = xfd[1][0]
    d = b - a
    dfia = xfd[0][2]

    c = diff(xfd.T[1][0:2]) - d * dfia

    eps = np.finfo(float).eps
    if c >= (5 * n * eps * b):
        a_1 = a - .5 * dfia * (d ** 2 / c)
        d *= 0.1
        alpha = min(array([max(array([(a + d), a_1])), b - d]))
    else:
        alpha = (a + b) / 2

    return alpha


def check_d(self, n, do):
    """
    Check given inverse Hessian.
    """
    d = do
    sd = len(d)

    if any(sd - n) != 0:
        print('D  should be a square matrix of size ' + n)

    # Check symmetry
    d_d = d - d.T
    n_dd = linalg.norm(d_d[:], inf)

    eps = np.finfo(float).eps

    if n_dd > 10 * eps * linalg.norm((d[:], inf)):
        print('Error: The given D0 is not symmetric')

    if n_dd == (d + d.T) / 2 and d == (d + d.T) / 2:
        return

    p = linalg.cholesky(d)
    return d


def softline(x, f, g, h, x1=None, m=None, n=None):
    """
    Soft line search: Find alpha = argmin_a{f(x+a*h)}
    """
    global fib, dfib
    n1 = n
    # Default return values
    h = h.reshape((1, len(h)))[0]

    g = g.reshape((1, len(g)))[0]
    alpha = 0
    fn = f
    gn = g
    neval = 0
    slrat = 1
    n = len(x)

    # Initial values

    dfi0 = dot(h, gn)

    if dfi0 >= 0:
        return alpha, fn, gn, neval, slrat

    fi0 = f
    slope0 = .05 * dfi0
    slopethr = .995 * dfi0
    dfia = dfi0
    stop = 0
    ok = 0
    neval = 0
    b = 1

    while stop == 0:
        bh = b * h
        bh = bh.reshape((len(h), 1))
        bh = (x + bh).reshape((1, len((x + bh))))

        fib, g = ica_mlf(bh, x=x1, m=m, n=n1)

        neval += 1
        g1 = []
        for i in g:
            for j in i:
                g1.append(j)
        g = array(g1)
        dfib = dot(g, h)

        if b == 1:
            slrat = dfib / dfi0
        fib = fib[0]

        fib_compare = (fi0 + (slope0 * b))

        if fib <= fib_compare:

            if dfib > abs(slopethr):
                stop = 1
            else:
                alpha = b
                fn = fib
                gn = g
                dfia = dfib
                ok = 1
                slrat = dfib / dfi0

                if (neval < 5) and (b < 2) and (dfib < slopethr):
                    b *= 2
                else:
                    stop = 1
        else:
            stop = 1

    stop = ok
    xfd = [[alpha, fn, dfia], [b, fib, dfib], [b, fib, dfib]]

    while stop == 0:
        c = interpolate(xfd, n=n)
        ch = c * h
        ch = ch.reshape((len(h), 1))
        ch = (x + ch).reshape((1, len((x + ch))))
        fic, g = ica_mlf(ch, x=x1, m=m, n=n1)
        neval += 1
        g1 = []
        for i in g:
            for j in i:
                g1.append(j)
        g = array(g1)
        xfd[2] = [c, fic, dot(g, h)]

        if fic < (fi0 + slope0 * c):
            xfd[0] = xfd[2]
            ok = 1
            alpha = c
            fn = fic
            gn = g
            slrat = xfd[2][2] / dfi0
        else:
            xfd[1] = xfd[1]
            ok = 0

        ok &= abs(xfd[2][2]) <= abs(slopethr)

        stop = ok | (neval >= 5) | (diff(xfd[0:1], n=2) <= 0)

    return alpha, fn, gn, neval, slrat


def ucminf(init_step, opts_2, opts_3, opts_4, w=None, x0=None, d0=None, x=None, m=None, n=None):
    """
    UCMINF BFGS method for unconstrained nonlinear optimization:
    Find  xm = argmin{f(x)} , where  x  is an n-vector and the scalar function F with gradient g (with elements
    g(i) = DF/Dx_i)
    """
    xpar = x
    # Check call
    opts0 = array([init_step, opts_2, opts_3, opts_4])
    n1 = n
    x, n, f, g, opts = check(x0, opts0, x=x, w=w, m=m, n=n)

    if d0 is not None:
        d = check_d(n, d0)
        fst = 0
    else:
        d = eye(n)
        fst = 1

    # Finish initialization
    k = 1
    kmax = opts_4
    neval = 1
    ng = linalg.norm(g, inf)
    delta = opts[0]

    x1 = []

    for i in x:
        for j in i:
            x1.append([j])
    x = array(x1)

    x_1 = x * ones((1, kmax + 1))

    b = [f, ng]
    zeros_array = zeros((3, 1))
    for i in zeros_array:
        b.append(i)
    b.append(delta)
    b = array(b)
    perf = b * ones((kmax + 1, 1))

    found = ng <= opts_2

    h = zeros(x.shape)

    nh = 0
    ngs = ng * ones((1, 3))

    while not found:
        xp = array(x)
        gp = array(g)
        fp = f
        nx = linalg.norm(x)
        ngs = [ngs[2:3], ng]
        negative_g = (-g.reshape((1, len(g))))
        h = dot(d, -g)
        nh = linalg.norm(h)
        red = 0

        if nh <= opts_3 * (opts_3 + nx):
            print('Entro en 1')
            found = True

        else:
            if fst > delta or nh > delta:
                h *= (delta / nh)
                nh = delta
                fst = 0
                red = 1

            k += 1

            al, f, g, dval, slrat = softline(x, f, g, h, x1=xpar, m=m, n=n1)

            if al < 1:
                delta *= .35
            elif red != 0 and (slrat > .7):
                delta *= 3

            h = h.reshape((len(h), 1))

            x += al * h

            neval = neval + dval
            ng = linalg.norm(g, inf)
            h = x - xp

            nh = linalg.norm(h)

            if nh == 0:
                print('Entro en 2')
                found = True
            else:
                gp = gp.reshape((1, len(gp)))
                y = g - gp
                h = h.reshape((1, len(h)))
                y = array(y[0])
                h = array(h[0])
                yh = dot(y, h)

                eps = np.finfo(float).eps

                if yh > (sqrt(eps) * nh * linalg.norm(y)):
                    y = y.reshape((1, len(y)))

                    v = dot(y, d)

                    v = v[0].reshape((1, len(v[0])))

                    yv = dot(y[0], v[0])

                    a = (1 + yv / yh) / yh
                    h = h.reshape((len(h), 1))
                    v = v[0].reshape((len(v[0]), 1))
                    w = ((a / 2) * h) - v / yh

                    d = d + dot(w, h.T) + dot(h, w.T)

                thrx = opts[2] * (opts[2] + linalg.norm(x))

                if ng <= opts[1]:
                    found = True

                elif nh <= thrx:
                    found = True

                elif neval >= kmax:
                    found = True

                else:
                    delta = max(delta, 2 * thrx)

    x_1 = x_1[:, 1:k]
    perf = perf[:, 1:k]
    return x, f, ng, nh, k - 1, neval, found, perf, d


def sort_array(array):
    sorted_array = sort(array)
    sorted_array = sorted_array[::-1]

    indx = []

    for i in range(len(array)):
        for j in range(len(array)):
            if sorted_array[i] == array[j]:
                indx.append(j)
                break
    return sorted_array, indx


def infomax_ica(s=None, a=None, u=None, ll=None, x=None, k=None, init_step=1, par_2=1e-4,
                par_3=1e-8,
                gradient=0, max_it=1000, info=None,
                limit=0.001, verbose=False, whitened=False, white_comp=None, white_parm=None,
                input_dim=None, dtype=None):
    """
    Input arguments:

    s -- Estimated source signals with variance scaled to one.

    a -- Estimated mixing matrix.

    u -- Principal directions of preprocessing PCA. If K (the number of sources) is equal to the number
         of observations then no PCA is performed and U=eye(K).

    ll -- Log likelihood for estimated sources.

    x -- Mixed Signals.

    k -- Number of source components. For K=0 (default) number of sources are equal to number of observations.
         For K < number of observations, SVD is used to reduce the dimension.

    init_step -- Expected length of initial step.

    gradient -- Gradient  ||g||_inf <= gradient.

    max_it -- Maximum number of iterations.

    info -- A dictionary  wPerformance information, dictionary with some elements.


    """

    # Algorithm parameter settings.

    max_nr_it = 1000

    # Scale X to avoid numerical problems.
    x = x
    x_org = x

    scale_x = max(array((max(x[:]), abs(min(x[:])))))

    x /= scale_x

    #Set number of source parameters

    if k is None:
        k = len(x[1])

    if (k > 0) and (k < len(x[1])):
        u, x = call_svd(x, k, 1)

    initial_step = init_step
    ucminf_opt_2 = par_2
    ucminf_opt_3 = par_3
    ucminf_opt_4 = max_it

    # Initialize variables

    m = x.shape[0]
    n = x.shape[1]

    w = eye(m)

    # Optimize

    w, f, ng, nh, k, neval, found, perf, d = ucminf(
        init_step, ucminf_opt_2, ucminf_opt_3, ucminf_opt_4, x0=w[:], x=x, w=w, m=m, n=n)

    w = w.reshape((m, m))

    # Estimates

    a = linalg.pinv(w)

    s = dot(w, x)

    # Sort components according to energy.

    a_var = array(diag(dot(a.T, a)) / m).reshape((len(array(diag(dot(a.T, a)) / m)), 1))
    s_var = array(diag(dot(s, s.T)) / n).reshape((len(array(diag(dot(s, s.T)) / n)), 1))
    v_s = var(s.T)
    sig = a_var * s_var
    sig = sig.reshape((1, len(sig)))[0]
    a1, indx = sort_array(sig)

    s = s[indx]

    # Scale back
    a = a * scale_x

    # Log likelihood

    ll = n * np.log(abs(linalg.det(linalg.inv(a)))) - sum(sum(np.log(np.cosh(s)))) - n * m * np.log(pi)

    return s
