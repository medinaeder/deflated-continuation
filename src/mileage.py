import numpy as np
from deflatedcontinuation import DeflatedContinuation as DC

class MileageObjective(DC):
    def __init__(self, problem, controls, params,has_trivial=True):
        super().__init__(problem(controls), params,has_trivial)

    def objective(self):
        cont = True
        self.run()
        obj = 0
        for s in self.solutions:
            p = s[0]
            for v in s[1]:
                stab = self.problem.stability(v,p)
                if cont: 
                    c = float(stab)
                else:
                    c = np.sign(stab)
                obj+=c
                  
            if self.has_trivial:
                x0 = np.zeros(self.problem.dim)
                stab = self.problem.stability(x0,p)
                if cont: 
                    c = float(stab)
                else:
                    c = np.sign(stab)
                obj+=c

        return obj


if __name__ == "__main__":
    from examples.pitchfork import Pitchfork
    import matplotlib.pyplot as plt
    xs = np.linspace(-1,1,21)
    for x in xs: 
        m = MileageObjective(Pitchfork, [x,1e-2], np.linspace(-1,2,101),False)
        plt.scatter(x,m.objective())

    m.plot_solutions()
    plt.show()
