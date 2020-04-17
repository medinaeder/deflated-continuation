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
        return np.array([1e-3,1e-3,0])

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
    b = 8/3
    s = 10

    problem = Lorenz(s,b)
    problem.solve_newton()
    import matplotlib.pyplot as plt
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
   
    import sys
    #sys.exit()
    sigs = np.linspace(5,25,100)
    fit =[]
    bs = np.linspace(1,10,100)
    for bb in bs:
        fit.append(problem(np.array([10,bb])))

    plt.scatter(bs, fit)
    plt.show()

        

    
