import numpy

DTYPE = numpy.int

numpy_dstack = numpy.dstack
numpy_select = numpy.select
numpy_roll = numpy.roll
numpy_minimum = numpy.minimum
numpy_zeros = numpy.zeros

def int_min(a, b):
    return a if a <= b else b

def int_min4(a, b, c, d):
    return int_min(int_min(a, b), int_min(c, d))

def solvemaze(corridors_in, mstart_in, maxiters=0, costs=None):
    M = mstart_in.shape[0]
    N = mstart_in.shape[1]
    D = mstart_in.shape[2]
    if maxiters is None or not maxiters:
        maxiters = 5 * (M + N)
    infinity = 2**20
    corridors = numpy_zeros(
            (M, N, D),
            dtype=DTYPE,
        ) + numpy_dstack((corridors_in, ) * D)
    mstart = numpy_zeros(
            (M, N, D),
            dtype=DTYPE,
        ) + (mstart_in == 1)
    m = numpy_zeros(
            (M, N, D),
            dtype=DTYPE,
        ) + infinity
    if costs is None:
        costs_r = numpy_zeros((M, N), dtype=DTYPE) + 1
    else:
        costs_r = numpy_zeros((M, N), dtype=DTYPE) + costs

    for i in range(maxiters):
        changed = 0
        done = True
        for x in range(0, M):
            for y in range(0, N):
                if corridors[x, y, 0]:
                    for z in range(D):
                        if mstart[x, y, z]:
                            m[x, y, z] = 1
                        else:
                            tmp = int_min4(
                                    m[x+1, y, z] if x < M - 1 else infinity,
                                    m[x-1, y, z] if x > 0 else infinity,
                                    m[x, y+1, z] if y < N - 1 else infinity,
                                    m[x, y-1, z] if y > 0 else infinity,
                                ) + costs_r[x, y]
                            if tmp != m[x, y, z]:
                                changed += 1
                                m[x, y, z] = tmp
                            elif tmp >= infinity:
                                done = False
        if changed == 0:
            m = numpy_select([corridors], [m], 0)
            if done:
                return m
            else:
                return None
    else:
        return None

try:
    import pyximport
except ImportError:
    pass
else:
    pyximport.install()
    from fastsolver import solvemaze
