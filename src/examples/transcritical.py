# Pitchfork Bifurcation Example
import numpy as np


class Transcritical:
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
        return np.array(p*u-b*u*u)+self.d

    def jacobian(self, u, p):
        b = self.b
        return np.array(p-2*b*u)

    def stability(self, u, p):
        return self.jacobian(u,p)

    def initial_guess(self):
        return np.array(0)

