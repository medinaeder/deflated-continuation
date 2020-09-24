import numpy as np
import scipy.optimize

class Lorenz:
    def __init__(self,controls):
        self.sigma = controls[0]
        self.b = controls[1]
        self.dim = 3

    def set_controls(self, controls):
        self.sigma = controls[0]
        self.b = controls[1]

    def __call__(self, x):
        self.sigma = x[0]
        self.b = x[1]
        self.solve_newton()
        fitness = self.objective()
        return fitness

    def residual(self, u ,r):
        x = u[0]
        y = u[1]
        z = u[2]

        sig = self.sigma
        b = self.b
        
        return np.array([sig*(y-x), r*x-y-x*z, -b*z+x*y])
    
    def residual_time(self,t, u ,r):
        x = u[0]
        y = u[1]
        z = u[2]

        sig = self.sigma
        b = self.b
        
        return np.array([sig*(y-x), r*x-y-x*z, -b*z+x*y])

    def jacobian(self, u, r):
        # Jacobian
        x = u[0]
        y = u[1]
        z = u[2]
        sig = self.sigma
        b = self.b
        return np.array([[-sig, sig, 0],[r-z,-1,-x],[y,x,-b]])

    def stability(self,u,r):
        # Larget Lyapunov Exponent
        J = self.jacobian(u,r)
        e, v = np.linalg.eig(J) 
        return max(e)

    def initial_guess(self):
        return [np.ones(3)*1e-8, np.ones(3)*-1e-8]

    def functional(self,u):
        return u[0]

if __name__== "__main__":
    import matplotlib.pyplot as plt
    b = 8/3
    s = 10

    # Plot the Bifurcation Diagram of the Lorenz System
    problem = Lorenz(s,b)
    problem.solve_newton()
    r =[]
    d =[]
    s =[]
    for x in problem.sol:
        r.append(x[0])
        d.append(x[1])
        s.append(x[2])
    
    r = np.array(r)
    d = np.array(d)
    s = np.array(s)
    print(problem.objective())
    plt.scatter(r, d[:,1], c = np.sign(s) ,vmin = -1, vmax = 1, cmap = "PiYG_r")
    plt.colorbar()
    plt.show()

    # Simulate the Lorenz System Starting from a random initial position
    # Choose a number of steps
    from scipy import integrate
    r0 = 1.1
    delta = 1e-3
    output = integrate.solve_ivp(problem.residual_time, t_span = (0,10), y0 = problem.initial_guess() ,args = (r0+delta,),  t_eval = np.linspace(0,10,101))
    y = output.y
    # Visualize this system
    #print(output)
    print(output.t)
    print(y.shape)
    plt.plot(output.t, y[0,:])
    plt.figure()
    plt.plot(output.t, y[1,:])
    plt.figure()
    plt.plot(output.t, y[2,:])
    plt.figure()
    plt.plot(y[0,:], y[1,:])

    plt.show()

   
    #import sys
    #sys.exit()
    sigs = np.linspace(5,25,100)
    fit =[]
    bs = np.linspace(1,10,100)
    for bb in bs:
        fit.append(problem(np.array([10,bb])))

    plt.scatter(bs, fit)
    plt.show()

        

    
