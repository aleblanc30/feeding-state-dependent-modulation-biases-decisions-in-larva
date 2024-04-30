from models import *
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root_scalar
import plotly.graph_objects as go
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
import os

def main():
    Model = BaseModel
    # Computations
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

            go.Scatter3d(x=[18.5,18.5],
                        y=[8.5,8.5],
                        z=[1.15,1.65],
                        line={'color':'darkblue',
                            'width':4},
                        mode='lines',
                        showlegend=False),

            go.Scatter3d(x=x_vert,
                        y=y_vert,
                        z=z_vert,
                        marker={'size':4, 'color':[to_rgb(c) for c in colors]},
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
    fig.write_image('figures/supplementary/explain_3d_plot_3d.pdf')
    fig.write_image('figures/supplementary/explain_3d_plot_3d.png')
    fig.show()

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
    fig.write_image('figures/supplementary/explain_3d_plot_3d_no_legend.pdf')
    fig.write_image('figures/supplementary/explain_3d_plot_3d_no_legend.png')
    fig.show()


    iLNa_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNa, 'excitatory', z_vert)
    iLNb_initializer = SimpleInitializer(iLNb=2)

    def r_max_initializer(model):
        model.r_max[Neuron.iLNa] = x_vert[0]
        model.update()
    r_max_initializer = ModelInitializer(r_max_initializer)

    def Hb_initializer(model):
        model.exogenous_current[Neuron.fbLNHb] = y_vert[0]
        model.update()
    Hb_initializer = ModelInitializer(Hb_initializer)

    results = Model.compute_states_on_grid([iLNb_initializer, r_max_initializer, Hb_initializer], [iLNa_iterator])

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    plt.figure()
    for iLNa, c in zip(iLNa_iterator, colors):
        res = results.results[(iLNa,)]
        plt.plot(res.y[Neuron.B1,:], res.y[Neuron.B2,:], c=c, lw=4)
    plt.legend(labels=[r"$w(mCh \rightarrow iLNb) = $"+f'{z:.3f}' for z in z_vert])
    plt.title('Trajectories in (B1, B2) space for the parameters on the vertical line')
    plt.savefig('figures/supplementary/explain_3d_plot_2d.pdf')
    plt.savefig('figures/supplementary/explain_3d_plot_2d.png')
    plt.show()

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    main()