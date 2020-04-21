# Pitchfork Bifurcation Example
import numpy as np


class Transcritical:
    def __init__(self,controls):
        self.b = controls[0]
        self.d = controls[1]
        self.dim = 1

    def __call__(self,x):
        print("Not Implemented Yet")

    def residual(self, u, p):
        """
        u: values
        p: parameter
        """
        b = self.b
        return np.array(p*u-b*u*u)+self.d

    def jacobian(self, u, p):
        b = self.b
        return np.array(p-2*b*u)

    def stability(self, u, p):
        return self.jacobian(u,p)

    def initial_guess(self):
        return [np.array(0.01)]
    
    def functional(self,u):
        return u
