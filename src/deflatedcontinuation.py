import numpy as np
import scipy.optimize 
# Add a try to import matplotlib
import matplotlib.pyplot as plt

# Deflate a single solution for now and run the exampples

class DeflatedContinuation:
    def __init__(self,problem, params):
        self.problem = problem
        self.params = params

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
                    #xn = scipy.optimize.newton(self.deflate, x, args = (p,problem.residual,[], tol = 1e-8)
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
                        xn = scipy.optimize.newton(self.deflate, x0[i], args = (p,problem.residual,xnew,), tol = 1e-8)
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
    
    def deflate(self, x, p, f, xstars, shift=1, power=2):
        #print(x.shape)
        #I = np.identity(x.shape[0])
        factor = 1;
        for xstar in xstars:
            denom = (np.linalg.norm(x-xstar))**power
            factor *= 1./denom + shift

        # deflate the trivial solution
        #denom = (np.linalg.norm(x))**power
        #factor *= 1./denom + shift
        return factor*f(x,p)

    def deflate_jacobian(self, x, p, f, xstars, shift=1, power=2):
        pass

    
    def plot_solutions(self):
        plt.figure()
        for s in self.solutions:
            p = s[0]
            for v in s[1]:
                stab = self.problem.stability(v,p)
                plt.scatter(p, self.problem.functional(v), c = np.sign(stab), vmin = -1, vmax = 1)


if __name__ == "__main__":
    from examples.pitchfork import Pitchfork
    from examples.saddle import Saddle
    from examples.transcritical import Transcritical

    problem = Transcritical(1e-2)
    params = np.linspace(-1, 2,101)
    df = DeflatedContinuation(problem,params)
    df.run()
    df.plot_solutions()
    
    problem = Pitchfork(1e-5)
    params = np.linspace(-1,2,101)
    df = DeflatedContinuation(problem,params)
    df.run()
    df.plot_solutions()

    problem = Saddle()
    params = np.linspace(-1,2,101)
    df = DeflatedContinuation(problem,params)
    df.run()
    df.plot_solutions()

    plt.show()



    #x0 = scipy.optimize.newton(problem.residual, np.array([0]), args = (1.01,))
    #print(x0)
    #x1 = scipy.optimize.newton(df.deflate, np.array([0]), args = (1.01,problem.residual,[x0],), tol = 1e-6)
    #print(x1)
    #x2 = scipy.optimize.newton(df.deflate, np.array([0]), args = (1.01,problem.residual,[x0,x1],), tol = 1e-6)
    #print(x2)
