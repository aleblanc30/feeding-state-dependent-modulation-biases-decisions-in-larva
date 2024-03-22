from models import *
import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go
from scipy.optimize import root_scalar
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

Model = BaseModel

def main():
    # Computations
    n_weights = 11

    iLNa_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNa, 'excitatory', np.linspace(.5,1.5,n_weights))
    iLNb_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNb, 'excitatory', np.linspace(1.5,2.5,n_weights))

    results = Model.compute_states_on_grid([], [iLNa_iterator, iLNb_iterator])

    # Plots

    def create_color_grid(N):
        j, i = np.meshgrid(np.arange(N), np.arange(N))
        return np.stack([1-j/N, i/N, .5*np.ones_like(i)], axis=2)

    def plot_color_grid(ax, it1, it2):
        i, j = np.meshgrid(it1, it2)
        i, j= i.flatten(), j.flatten()
        mi, Mi = np.min(i), np.max(i)
        mj, Mj = np.min(j), np.max(j)
        ax.scatter(i,j, s=15, c=[(1-(y-mj)/(Mj-mj), (x-mi)/(Mi-mi), .5) for x, y in zip(i, j)])
        ax.set_xticks([mi,Mi])
        ax.set_yticks([mj,Mj])

    def plot_phase_diagram(results, n_weights):
        plt.figure()
        plt.axis('equal')
        
        color_grid = create_color_grid(n_weights)
        for (i,iLNa), (j,iLNb) in itertools.product(enumerate(iLNa_iterator), enumerate(iLNb_iterator)):
            plt.plot(results[(iLNa, iLNb)].y[Neuron.B1,:], results[(iLNa, iLNb)].y[Neuron.B2,:], c=color_grid[i,j])

        subax = plt.gca().inset_axes([0.1,.65,.25,.25])
        plot_color_grid(subax, iLNa_iterator, iLNb_iterator)
        subax.axis('equal')
        subax.set_xlabel('iLNa')
        subax.set_ylabel('iLNb')
        plt.savefig('figures/supplementary/consistency.pdf')
        plt.savefig('figures/supplementary/consistency.png')

    plot_phase_diagram(results, n_weights)

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    main()
    os.system('''cp -R "figures/." "/home/alexandre/Insync/blanc.alexandre.perso@gmail.com/Google Drive - Shared with me/For_submission/Figures/plots_from_alexandre/final"''')