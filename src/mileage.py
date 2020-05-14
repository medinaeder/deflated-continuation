import numpy as np
from deflatedcontinuation import DeflatedContinuation as DC

class MileageObjective(DC):
    def __init__(self, problem, controls, params,has_trivial=True):
        super().__init__(problem(controls), params,has_trivial)
        self.dimension = len(controls)

    def objective(self,controls):
        self.problem.set_controls(controls)
        cont = False
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
                if c<0:
                    obj+=c
                  
            if self.has_trivial:
                x0 = np.zeros(self.problem.dim)
                stab = self.problem.stability(x0,p)
                if cont: 
                    c = float(stab)
                else:
                    c = np.sign(stab)
                if c<0:
                    obj+=c
        
        #FIXME: Need to clear the previous solutions. To not double count yadada 
        self.problem.solutions =[]
        return obj


if __name__ == "__main__":
    from examples.pitchfork import Pitchfork
    import matplotlib.pyplot as plt
    xs = np.linspace(1,-1,21)
    for x in xs: 
        m = MileageObjective(Pitchfork, [x,1e-2], np.linspace(-1,2,101),False)
        plt.scatter(x,m.objective(), c = 'k')

    m.plot_solutions()
    plt.show()

    m = MileageObjective(Pitchfork, [-1,1e-2], np.linspace(2,-1,101),False)
    m.run()
    m.plot_solutions()
    plt.show()

