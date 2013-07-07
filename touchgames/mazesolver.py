def solvemaze(corridors, start_point):
    active = [start_point]
    while active:
        this_point = x, y = active.pop()
        corridors[this_point] = False
        for next_point in ((x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)):
            if corridors[next_point]:
                active.append(next_point)
        #print corridors
    if corridors.any():
        return False
    return True
