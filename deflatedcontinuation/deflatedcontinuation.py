import numpy as np
import scipy.optimize 
import matplotlib.pyplot as plt

from .newtonsolver import newtonsolver, newtonsolver1d

class DeflatedContinuation:
    def __init__(self,problem, params, has_trivial = True, tol = 1e-6):
        self.problem = problem
        self.params = params
        self.has_trivial = has_trivial
        self.shift = 1
        self.power = 2
        self.tol = tol

    def run(self):
        problem = self.problem
        if problem.dim==1:
            newtonsolve = newtonsolver1d
        else:
            newtonsolve = newtonsolver
        # What is the structure of the solutions
        solutions = []
        
        # Solutions that were found in the previous loop
        prev_sol = []
        for i,p in enumerate(self.params):
            if not prev_sol: # Check to see if we are starting at the first point
                x0 = problem.initial_guess()
            else:
                x0 = prev_sol
           
            # TODO: Add a flag to include the offspring
            # Do we use our locally interpolated information
            # use_parent 
            # for sol in parent_sols[i]
            #   x0.append(sol[i])


            xnew = []
            # Continuation Step
            for x in x0:
                if self.has_trivial:
                    #xn = scipy.optimize.root(self.deflate, x, jac = self.deflate_jacobian, args = (p,[],), tol=self.tol, method = 'krylov')
                    xn = newtonsolve(self.deflate, x, jac = self.deflate_jacobian, args = (p,[],), atol=self.tol)
                else:
                    #xn = scipy.optimize.root(problem.residual, x,jac = self.problem.jacobian,  args = (p,) tol=self.tol)
                    xn = newtonsolve(problem.residual, x,jac = self.problem.jacobian,  args = (p,), atol=self.tol)
                if xn.success:
                    xnew.append(xn.x)

            # Branch Discovery Step
            num_sols = len(x0)
            for i in range(num_sols):
                discovered = True
                while discovered:
                    #xn = scipy.optimize.root(self.deflate, x0[i], jac = self.deflate_jacobian,  args = (p,xnew,), tol=self.tol, method = 'krylov')
                    xn = newtonsolve(self.deflate, x0[i], jac = self.deflate_jacobian,  args = (p,xnew,),atol=self.tol, max_it=20)
                    if xn.success:
                        xnew.append(xn.x)
                    else: 
                        discovered = False
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

        return eta*jac + np.outer(R,deta)

    def refine(self):
        # The idea is that we solve with a large tolerance
        # and then we refine the deflated solutions
        # Iterate through our solutions and use those as the initial guess to our problem
        refined_solutions = []
  
        for s in self.solutions:
            p = s[0] 
            new_set = []
            for x in s[1]:
                xn = newtonsolve(self.problem.residual, x,jac = self.problem.jacobian,  args = (p,), atol=self.tol)
                if xn.success:
                    new_set.append(xn.x)
            refined_solutions.append([p,new_set])
        self.refined_solutions = refined_solutions
        return

    def plot_solutions(self, refined = False):
        plt.figure()
        cont = False
        if refined:
            sols = self.refined_solutions
        else:
            sols = self.solutions
        for s in sols:
            p = s[0]
            for v in s[1]:
                stab = self.problem.stability(v,p)
                if cont: 
                    c = float(stab)
                else:
                    c = np.sign(stab)
                plt.scatter(p, self.problem.functional(v), c = c, vmin = -1, vmax = 1, cmap = "cividis")

            if self.has_trivial:
                x0 = np.zeros(self.problem.dim)
                stab = self.problem.stability(x0,p)
                if cont: 
                    c = float(stab)
                else:
                    c = np.sign(stab)
                plt.scatter(p, self.problem.functional(x0), c = c, vmin = -1, vmax = 1, cmap = "cividis")




