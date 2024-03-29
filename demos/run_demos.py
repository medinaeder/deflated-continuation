import numpy as np
from deflatedcontinuation import DeflatedContinuation
import matplotlib.pyplot as plt

def run_perturbed(pert):
    from pitchfork import Pitchfork
    from saddle import Saddle
    from transcritical import Transcritical
    from lorenz import Lorenz


    problem = Lorenz([10, 8./3])
    params = np.linspace(0.1, 30, 201)
    df = DeflatedContinuation(problem,params,True)
    df.run()
    df.plot_solutions()
    

    problem = Transcritical([1.,pert])
    params = np.linspace(-1, 2,101)
    df = DeflatedContinuation(problem,params,False)
    df.run()
    df.plot_solutions()


    problem = Pitchfork([1,pert])
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
    return

def run_perfect():
    from pitchfork import Pitchfork
    from saddle import Saddle
    from transcritical import Transcritical
    from lorenz import Lorenz
    
    problem = Lorenz([10, 8./3])
    params = np.linspace(0.1, 30, 101)
    #params = np.linspace(1.1,1.2, 2)
    df = DeflatedContinuation(problem,params, True, 1e-8)
    df.run()
    #print("refining")
    #df.refine()
    #import IPython; IPython.embed()
    df.plot_solutions(False)
    #plt.show()
    #import sys; sys.exit()
    

    problem = Transcritical([1.,0])
    params = np.linspace(-1, 2,101)
    df = DeflatedContinuation(problem,params,True, 1e-6)
    df.run()
    df.plot_solutions()


    problem = Pitchfork([1.,0])
    params = np.linspace(-1,2,101)
    df = DeflatedContinuation(problem,np.flip(params),True, 1e-6)
    df.run()
    df.plot_solutions()

    problem = Saddle()
    params = np.linspace(-1,2,101)
    df = DeflatedContinuation(problem,params,False, 1e-6)
    df.run()
    df.plot_solutions()

    plt.show()
    return

if __name__ == "__main__":
    #run_perfect()
    run_perturbed(1e-5)
