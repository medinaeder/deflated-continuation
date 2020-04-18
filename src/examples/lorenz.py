import numpy as np
import scipy.optimize

class Lorenz:
    def __init__(self,sig,b):
        self.sigma = sig 
        self.b = b
        self.dim = 3

    def __call__(self, x):
        self.sigma = x[0]
        self.b = x[1]
        self.solve_newton()
        fitness = self.objective()
        return fitness

    def residual(self, u ,r):
        # u is the solution, r is the parameter
        # RHS
        x = u[0]
        y = u[1]
        z = u[2]

        sig = self.sigma
        b = self.b
        
        return np.array([sig*(y-x), r*x-y-x*z, -b*z+x*y])
    
    def residual(self,t, u ,r):
        # u is the solution, r is the parameter
        # RHS
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
        J = self.jacobian(u,r)
        e, v = np.linalg.eig(J) 
        # Return the largest eigenvalue of the system
        return max(e)
    
        
    def solve_newton(self):
        rmax = 30.0
        r = np.linspace(0.0,rmax, 201)
        sols = []
        x = np.zeros(3)
        for i in range(len(r)*0):
            ri = r[i]
            try:
                x = scipy.optimize.newton(self.f, x, args=(ri,),tol=1e-6)
            except:
                print("F1", self.b,self.sigma)
            s = self.stability(x,ri)
            sols.append([ri, x,s]) 

        b =self.b

        r = np.linspace(1.0,rmax, 201)
        t = np.sqrt(b*(r[0]-1))
        x = np.array([t,t,r[0]-1])
        for i in range(len(r)):
            ri = r[i]
            t = np.sqrt(b*(ri-1))
            x = np.array([t,t,ri-1])
            try:
                x = scipy.optimize.newton(self.f, x, args=(ri,),tol=1e-6)
            except:
                print("F2", self.b,self.sigma)
            s = self.stability(x,ri)
            sols.append([ri, x,s]) 

        r = np.linspace(1.0,rmax, 201)
        for i in range(len(r)):
            ri = r[i]
            t = np.sqrt(b*(ri-1))
            x = np.array([-t,-t,ri-1])
            try:
                x = scipy.optimize.newton(self.f, x, args=(ri,),tol=1e-6)
            except:
                print("F3", self.b,self.sigma)
            s = self.stability(x,ri)
            sols.append([ri, x,s]) 
        self.sol = sols

    def initial_guess(self):
        b = self.b
        t = np.sqrt(b*(1.1-1))
        x = np.array([-t,-t,1.1-1])
        return x

    def objective(self):
        # maximize the number of stable points
        s = []
        for x in self.sol:
            s.append(x[2])

        #s = np.sign(np.array(s))
        s = np.array(s)
        return -np.sum(np.real(s))

    def functional(self,u):
        return u[0]

        

# First I want to plot the bifurcation diagram
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
    output = integrate.solve_ivp(problem.residual, t_span = (0,10), y0 = problem.initial_guess() ,args = (r0+delta,),  t_eval = np.linspace(0,10,101))
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
    #sigs = np.linspace(5,25,100)
    #fit =[]
    #bs = np.linspace(1,10,100)
    #for bb in bs:
    #    fit.append(problem(np.array([10,bb])))

    #plt.scatter(bs, fit)
    #plt.show()

        

    
