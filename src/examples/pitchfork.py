# Pitchfork Bifurcation Example
import numpy as np


class Pitchfork:
    def __init__(self,delta):
        self.b = 1
        self.d = delta

    def __call__(self,x):
        print("Not Implemented Yet")

    def residual(self, u, p):
        """
        u: values
        p: parameter
        """
        b = self.b
        return np.array(p*u-b*u*u*u)+self.d

    def jacobian(self, u, p):
        b = self.b
        return np.array(p-3*b*u*u)

    def stability(self, u, p):
        return self.jacobian(u,p)

    def initial_guess(self):
        return np.array(0)

    def functional(self,u):
        return u

if __name__ == "__main__":
    import scipy.optimize
    import matplotlib.pyplot as plt
    u0 = 0 
    ps = np.linspace(-1,1,22)
    u = []
    stab = []
    problem = Pitchfork()
    for a in ps:
        sol = scipy.optimize.newton(problem.residual, u0, problem.jacobian,  args=(a,))
        u.append(sol)
        stab.append(problem.stability(sol,a))

    plt.scatter(ps,u, c = np.sign(stab), vmin = -1, vmax = 1, cmap = "PiYG_r")
    plt.colorbar()
    plt.show()

        
