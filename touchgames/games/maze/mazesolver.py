import numpy

def solvemaze(corridors, mstart, maxiters=None):
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
    """
    M, N, D = mstart.shape
    if not maxiters:
        maxiters = 11 * (M + N)
    infinity = 2**30
    corridors = numpy.dstack([corridors] * D)
    m = numpy.select([mstart], [mstart], infinity)
    for i in range(maxiters):
        m = numpy.select(
                [mstart == 1, corridors],
                [mstart, m],
                infinity,
            )
        m = numpy.minimum(
                numpy.minimum(numpy.roll(m, 1, 0), numpy.roll(m, -1, 0)),
                numpy.minimum(numpy.roll(m, 1, 1), numpy.roll(m, -1, 1)),
            ) + 1
        m = numpy.select([corridors], [m], 0)
        theMax = m.max()
        if theMax < infinity:
            return m
    else:
        print "Maxed out"
        return None

try:
    import pyximport
    pyximport.install()
    from touchgames.games.maze.fastsolver import solvemaze
except ImportError:
    print 'WARNING: Fast maze solver could not be loaded!'
    raise
