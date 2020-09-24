# My own newton solver
# I just want to do a standard newton solve
# Maybe with a line search?
import numpy as np

# Similar handles as scipy OptimizeResult
class NewtonSolution:
    def __init__(self, x, success):
        self.success = success
        self.x = x

def newtonsolver1d(fun, x0, jac, args=(), atol=1e-8, rtol=1e-12, max_it=100):
    # Tolerances
    # stol:     step size changse
    # atol:     norm of the residual
    # rtol:     relative norm change
    x = np.copy(x0)
    R = fun(x, *args)
    stol =  1e-10
    rnorm0 = np.linalg.norm(R)
    count = 0
    while count < max_it:
        J = jac(x,*args)
        #e,v = np.linalg.eigh(J)
        if J.shape==(1,1):
            J = J[0,0]
        dx = -R/J
        count+=1
        # TODO: Add a linesearch
        norm_dx = np.linalg.norm(dx)
        alpha = 1.0
        x+=alpha*dx
        R = fun(x,*args)
        rnorm = np.linalg.norm(R)
        if norm_dx < stol or rnorm < np.max((rtol*rnorm0, atol)): 
            print("Number of Iterations", count)
            if norm_dx< stol:
                print("Step_Size Reached", norm_dx)
            if rnorm < atol:
                print("Absolute Tolerance Reached", rnorm)
            else:
                print("Relative Tolerance reached", rtol*rnorm0)
            sol = NewtonSolution(x,True)
            return sol
            
    
    return NewtonSolution(x0, False)

def newtonsolver(fun, x0, jac, args=(), atol=1e-8, rtol=1e-12, max_it=100):
    # Tolerances
    # stol:     step size changse
    # atol:     norm of the residual
    # rtol:     relative norm change
    x = np.copy(x0)
    R = fun(x, *args)
    stol =  1e-10
    rnorm0 = np.linalg.norm(R)
    count = 0
    while count < max_it:
        J = jac(x,*args)
        #e,v = np.linalg.eigh(J)
        dx = np.linalg.solve(J, -R)
        count+=1
        # TODO: Add a linesearch
        norm_dx = np.linalg.norm(dx)
        alpha = 1.0
        x+=alpha*dx
        R = fun(x,*args)
        rnorm = np.linalg.norm(R)
        if norm_dx < stol or rnorm < np.max((rtol*rnorm0, atol)): 
            print("Number of Iterations", count)
            if norm_dx< stol:
                print("Step_Size Reached", norm_dx)
            if rnorm < atol:
                print("Absolute Tolerance Reached", rnorm)
            else:
                print("Relative Tolerance reached", rtol*rnorm0)
            sol = NewtonSolution(x,True)
            return sol
            
    
    return NewtonSolution(x0, False)
    

