import numpy as np
from bifcma import CMAES
# This will load CMAES



# Import the Lorenz System
from examples.lorenz import Lorenz

# Import the mileage objective
from mileage import MileageObjective



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    control_dim = 2
    x = np.array([10., 8./3])
    pr = np.linspace(0.1,30,101)
    problem = MileageObjective(Lorenz, x,pr, has_trivial = True)
    problem.problem.set_controls(x)
    problem.run()
    problem.plot_solutions()
        
    import time;
    start = time.time()
    cma = CMAES(problem, x, 1,maxfevals=200)
    cma.optimize()
    print("Time Elapsed", time.time()-start)
    print(cma.best_solution)
    
    xstar = cma.best_solution
   
    problem.problem.set_controls(xstar)
    problem.run()
    problem.plot_solutions()
    plt.show()

    xs = np.array(cma.xs)
    fs = np.array(cma.fs)
    plt.scatter(xs[:,0], xs[:,1], c = fs)
    plt.colorbar()
    plt.show()
    
    
    

 
