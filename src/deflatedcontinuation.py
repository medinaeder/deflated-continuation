import numpy as np
import scipy.optimize 
# Add a try to import matplotlib
import matplotlib.pyplot as plt

# Deflate a single solution for now and run the exampples

class DeflatedContinuation:
    def __init__(self,problem, params, has_trivial = True):
        self.problem = problem
        self.params = params
        self.has_trivial = has_trivial
        self.shift = 1
        self.power = 2

    def run(self):
        problem = self.problem
        # What is the structure of the solutions
        solutions = []
        
        # Solutions that were found in the previous loop
        prev_sol = []
        for i,p in enumerate(self.params):
            if not prev_sol: # Check to see if we are starting at the first point
                x0 = [problem.initial_guess()]
            else:
                x0 = prev_sol
            xnew = []
            # Continuation Step
            for x in x0:
                try:
                    if self.has_trivial:
                        xn = scipy.optimize.newton(self.deflate, x, args = (p,[],), tol = 1e-8)
                    else:
                        xn = scipy.optimize.newton(problem.residual, x, problem.jacobian, args = (p,), tol = 1e-8)
                    xnew.append(xn)
                except: 
                    # FixMe to return a ConvergenceError
                    #print("Failed to Converge")
                    pass

            # Branch Discovery Step
            num_sols = len(x0)
            
            for i in range(num_sols):
                discovered = True
                while discovered:
                    try:
                        xn = scipy.optimize.newton(self.deflate, x0[i], self.deflate_jacobian, args = (p,xnew,), tol = 1e-8)
                        xnew.append(xn)
                        #print(xnew)
                    except: 
                        # FIXME: Return Convergence Error
                        #print("Failed to converge")
                        discovered=False
            # record solutions for parameter
            prev_sol = xnew
            solutions.append([p,xnew])

        self.solutions = solutions
    
    def deflate(self, x, p, roots):
        f = self.problem.residual
        shift = self.shift
        power = self.power
        factor = 1.0
        for root in roots:
            d = x - root
            normsq = np.dot(d,d)
            factor *= normsq**(-power/2.0) + shift

        # deflate the trivial solution
        if self.has_trivial:
            normsq = np.dot(x,x)
            factor *= normsq**(-power/2.0) + shift
        return factor*f(x,p)

    def deflate_jacobian(self, x, p, roots):
        shift = self.shift
        power = self.power
        problem = self.problem
        R = problem.residual(x,p)
        jac = problem.jacobian(x,p)
        
        # Adapted from PEF defcon operator deflation
        if len(roots) < 1:
            return jac

        factors = []
        dfactors = []
        normsqs = [] # norm squared
        dnormsqs = [] # norm squared

        for root in roots:
            d = x-root
            normsqs.append(np.dot(d,d))
            dnormsqs.append(2*d)
        
        if self.has_trivial:
            normsqs.append(np.dot(x,x))
            dnormsqs.append(2*x)

        for normsq in normsqs:
            factor = normsq**(-power/2.0) + shift
            dfactor = (-power/2.0)*normsq**((-power/2.0)-1.0)

            factors.append(factor)
            dfactors.append(dfactor)

        eta = np.prod(factors)
        deta = np.zeros_like(x)
        for (factor, dfactor, dnormsq) in zip(factors, dfactors, dnormsqs):
            deta+=(eta/factor)*dfactor*dnormsq

        return eta*jac + np.outer(deta,R)

    
    def plot_solutions(self):
        plt.figure()
        for s in self.solutions:
            p = s[0]
            for v in s[1]:
                stab = self.problem.stability(v,p)
                plt.scatter(p, self.problem.functional(v), c = np.sign(stab), vmin = -1, vmax = 1, cmap = "cividis")

            if self.has_trivial:
                x0 = np.zeros(self.problem.dim)
                stab = self.problem.stability(x0,p)
                plt.scatter(p, self.problem.functional(x0), c = np.sign(stab), vmin = -1, vmax = 1, cmap = "cividis")




if __name__ == "__main__":
    from examples.pitchfork import Pitchfork
    from examples.saddle import Saddle
    from examples.transcritical import Transcritical

    problem = Transcritical(1e-1)
    #problem = Transcritical(0)
    params = np.linspace(-1, 2,101)
    df = DeflatedContinuation(problem,params,False)
    df.run()
    df.plot_solutions()


    problem = Pitchfork(1e-1)
    params = np.linspace(-1,2,101)
    df = DeflatedContinuation(problem,params,False)
    df.run()
    df.plot_solutions()

    problem = Saddle()
    params = np.linspace(-1,2,101)
    df = DeflatedContinuation(problem,params,False)
    df.run()
    df.plot_solutions()

    plt.show()



    #x0 = scipy.optimize.newton(problem.residual, np.array([0]), args = (1.01,))
    #print(x0)
    #x1 = scipy.optimize.newton(df.deflate, np.array([0]), args = (1.01,problem.residual,[x0],), tol = 1e-6)
    #print(x1)
    #x2 = scipy.optimize.newton(df.deflate, np.array([0]), args = (1.01,problem.residual,[x0,x1],), tol = 1e-6)
    #print(x2)
