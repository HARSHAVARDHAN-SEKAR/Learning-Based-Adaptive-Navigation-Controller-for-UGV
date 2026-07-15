from planners.planners import astar as _astar
def theta_star(m, seed=0):
    return _astar(m, theta_variant=True)
