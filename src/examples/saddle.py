# Saddle Point 
import numpy as np


class Saddle:

    def __call__(self,x):
        print("Not Implemented Yet")

    def residual(self, u, p):
        """
        u: values
        p: parameter
        """
        return np.array(p-u*u)

    def jacobian(self, u, p):
        return np.array(-2*u)

    def stability(self, u, p):
        return self.jacobian(u,p)

    def initial_guess(self):
        return np.array(0)

