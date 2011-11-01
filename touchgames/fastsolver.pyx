import numpy
cimport numpy
cimport cython

DTYPE = numpy.int
ctypedef numpy.int_t DTYPE_t

cdef numpy_dstack = numpy.dstack
cdef numpy_select = numpy.select
cdef numpy_roll = numpy.roll
cdef numpy_minimum = numpy.minimum
cdef numpy_zeros = numpy.zeros

cdef inline int int_min(int a, int b):
    return a if a <= b else b

cdef inline int int_min4(int a, int b, int c, int d):
    return int_min(int_min(a, b), int_min(c, d))

def solvemaze(
        numpy.ndarray corridors_in not None,
        numpy.ndarray mstart_in not None,
        int maxiters=0,
        numpy.ndarray costs=None,
    ):
    """Solve a MxN maze with respect to D entry points

    corridors is a MxN matrix with true values where the maze is passable
    mstart is a MxNxD matrix with true values for entry points
    maxiters is the maximum number of iterations, or equivalently the
        maximum path length from an entry point to any other point in the
        maze. If None or 0, it's set to a default that marks some technically
        correct mazes as unsolvable (for real-time use in a game).

    Returns None if the maze is "invalid", i,e, at least corridor is not
        accessible from one of the entry points in maxiters steps or less.

    Otherwise, returns a MxNxD matrix where each MxN slice contains
        the length of the shortest path from the corresponding entry point.
        Entries corresponding to walls are undefined.

    Walls are assumed around the whole perimeter of the maze.
    """
    cdef int M = mstart_in.shape[0]
    cdef int N = mstart_in.shape[1]
    cdef int D = mstart_in.shape[2]
    if maxiters is None or not maxiters:
        maxiters = 5 * (M + N)
    cdef int infinity = 2**20
    cdef numpy.ndarray[DTYPE_t, ndim=3] corridors = numpy_zeros(
            (M, N, D),
            dtype=DTYPE,
        ) + numpy_dstack((corridors_in, ) * D)
    cdef numpy.ndarray[DTYPE_t, ndim=3] mstart = numpy_zeros(
            (M, N, D),
            dtype=DTYPE,
        ) + (mstart_in == 1)
    cdef numpy.ndarray[DTYPE_t, ndim=3] m = numpy_zeros(
            (M, N, D),
            dtype=DTYPE,
        ) + infinity
    cdef numpy.ndarray[DTYPE_t, ndim=2] costs_r
    if costs is None:
        costs_r = numpy_zeros((M, N), dtype=DTYPE) + 1
    else:
        costs_r = numpy_zeros((M, N), dtype=DTYPE) + costs
    cdef int i, x, y, z
    cdef int done, changed, tmp

    with cython.boundscheck(False):
        with cython.wraparound(False):
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
