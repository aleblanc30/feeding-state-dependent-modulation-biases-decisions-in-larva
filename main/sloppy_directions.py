from models import *
import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go
from scipy.optimize import root_scalar
from matplotlib.lines import Line2D


def supplementary():
    # Computations
    Model = CombinedModel
    n_weights = 20
    iLNb_initializer = SimpleInitializer(iLNb=2)
    r_max_iterator = ParameterIterator(np.linspace(17,20,n_weights), lambda model, r_max: model.set_r_max(Neuron.iLNa, r_max))
    Hb_iterator = ParameterIterator(np.linspace(0,20,n_weights), lambda model, current : model.set_exogenous_current(Neuron.fbLNHb, current))
    ths = Model.compute_thresholds_on_grid([iLNb_initializer], [Hb_iterator, r_max_iterator], bracket=[.5,2.5])
    thresholds = np.empty((n_weights, n_weights))
    for ids, tup in zip(itertools.product(range(n_weights), range(n_weights)), itertools.product(Hb_iterator, r_max_iterator)):
        thresholds[ids] = ths.results[tup]
    
    rmax, inputcurr = np.meshgrid(r_max_iterator, Hb_iterator)
    th = RegularGridInterpolator((r_max_iterator.iterator, Hb_iterator.iterator), thresholds.transpose())

    def solve_for_y(x,z):
      def aux(y):
            return th([x,y])-z
      return root_scalar(aux, method='brentq', bracket=[0,20], x0=10).root
    print(solve_for_y(18.65, 1.35))
    x_contour = [18.65,19.1,19.65]
    z_contour = 3*[1.35]
    y_contour = [solve_for_y(x,z) for x, z in zip(x_contour, z_contour)]
    print(y_contour)

    x_vert, y_vert, th_vert = 4*[18.5], 4*[8.5], th([18.5,8.5]).item()
    z_vert = [th_vert-0.1, th_vert-0.01, th_vert+0.01, th_vert+0.1]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    data = [
            go.Surface(x=rmax, 
                    y=inputcurr, 
                    z=thresholds,
                    contours={'z' : {'show': True,
                                        'start':1.2,
                                        'end':1.7,
                                        'size': 0.075}},
                    opacity=0.7,
                    colorbar={'title':'behavior threshold'}),
        
            go.Scatter3d(x=x_contour,
                        y=y_contour,
                        z=z_contour,
                        marker={'size':4, 'color':'red'},
                        mode='markers',
                        showlegend=False)
            
    ]

    layout = {
        'scene': {
                'camera':{
                    'eye':{'x':.39, 'y':-2.47, 'z':.65}
                    },
                'xaxis_title': 'r_max for iLNa',
                'yaxis_title': 'Input current to Hb',
                'zaxis_title': 'Affinity between input and iLNa',

                },
                'autosize':False,
                'width':700,
                'height':700
    }
    fig = go.Figure(data=data, layout=layout)

    fig.write_image('figures/supplementary/sloppy_directions_thresholds.pdf')
    fig.write_image('figures/supplementary/sloppy_directions_thresholds.png')

    layout = {
        'scene': {
                'camera':{
                    'eye':{'x':.39, 'y':-2.47, 'z':.65}
                    },
                'xaxis_title': '',
                'yaxis_title': '',
                'zaxis_title': '',
                },
                'autosize':False,
                'width':700,
                'height':700
    }
    fig = go.Figure(data=data, layout=layout)

    fig.write_image('figures/supplementary/sloppy_directions_thresholds_no_legend.pdf')
    fig.write_image('figures/supplementary/sloppy_directions_thresholds_no_legend.png')

    ###########################################################
    iLNb_initializer = SimpleInitializer(iLNb=2)
    iLNa_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNa, 'excitatory', np.linspace(.5,1.5,10))

    def tuple_iterator(model, param):
        rmax, input_current = param
        model.r_max[Neuron.iLNa] = rmax
        model.exogenous_current[Neuron.fbLNHb] = input_current
        model.update()
    tuple_iterator = ParameterIterator(list(zip(x_contour, y_contour)), tuple_iterator)

    results = Model.compute_states_on_grid([iLNb_initializer], [tuple_iterator, iLNa_iterator])
    norm=Normalize(min(iLNa_iterator), max(iLNa_iterator))
    cmap = get_cmap('magma')
    linestyles = ['-', '--', ':']
    plt.figure(figsize=(12,9))
    for t, ls in zip(tuple_iterator, linestyles):
        for iLNa in iLNa_iterator:
            res = results.results[(t,iLNa)]
            plt.plot(res.y[Neuron.B1,:], res.y[Neuron.B2,:], c=cmap(norm(iLNa)), ls=ls, lw=4)

    plt.legend(handles=[Line2D([],[], c='k', ls=ls, lw=4) for ls in linestyles],
            labels=[r'$r_{max}^{iLNa}$ = '+f'{r}, \n'+r'$i_{Hb}$ = '+f'{i:.2f}' for (r,i) in tuple_iterator], handlelength=3)
    plt.colorbar(ScalarMappable(norm, cmap), label=r'$w(Mch \rightarrow iLNa)$')
    plt.savefig('figures/supplementary/compare_dynamics_at_same_thresholds.pdf')
    plt.savefig('figures/supplementary/compare_dynamics_at_same_thresholds.png')

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    supplementary()