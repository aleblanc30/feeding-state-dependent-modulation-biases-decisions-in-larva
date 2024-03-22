from models import *
import os
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go

def neuron_activity(Model, folder, suffix):

    n_weights = 12

    iLNa_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNa, "excitatory", np.linspace(.5,1.5,n_weights))
    iLNb_initializer = SimpleInitializer(iLNb=2)
    internalStateIterator = Model.state_iterator

    results = Model.compute_states_on_grid([iLNb_initializer], [internalStateIterator, iLNa_iterator])

    cmap = colormaps['plasma']
    norm = Normalize(vmin=min(iLNa_iterator), vmax=max(iLNa_iterator))

    fig = plt.figure(figsize=(10,8))
    for iLNa in iLNa_iterator:
        r_fed = results[("fed", iLNa)]
        s_fed = results[("sucrose", iLNa)]
        plt.plot(r_fed.y[Neuron.B1], r_fed.y[Neuron.B2], color=cmap(norm(iLNa)), ls='-', lw=4)
        plt.plot(s_fed.y[Neuron.B1], s_fed.y[Neuron.B2], color=cmap(norm(iLNa)), ls='--', lw=4)

    plt.legend(handles=[Line2D([0],[0],ls='-',c='k', lw=4), Line2D([0],[0], ls='--', c='k', lw=4)], labels=['fed', 'sucrose'], loc='upper left', handlelength=3)
    plt.colorbar(ScalarMappable(norm, cmap), label=r'$w_{iLNa}$', location='right')
    plt.xlabel('B1 activity')
    plt.ylabel('B2 activity')
        
    fig.suptitle('Comparing neuron activity between fed and sucrose state, model '+suffix)
    plt.savefig(os.path.join('figures/neuron_activity_fed_vs_sucrose', folder, f'fed_vs_sucrose_neuron_activities_{suffix}.pdf'))
    # plt.show()

def phase_diagram(Model, folder, suffix):
    n_weights = 100
    if Model == InputCurrentHbModel:
        iLNb_initializer = SimpleInitializer(iLNb=2)
        iLNa_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNa, 'excitatory', np.linspace(1.1,1.4,n_weights))
        Hb_iterator = ParameterIterator(np.linspace(0,20,n_weights), lambda model, current : model.set_exogenous_current(Neuron.fbLNHb, current))
        results = Model.compute_states_on_grid([iLNb_initializer], [iLNa_iterator, Hb_iterator])
        results.plot_ratio_diagram()
        plt.xlabel('affinity of input signal to iLNa (AU)')
        plt.ylabel('Hb exogenous input current (AU)')
    if Model == DecreasedRmaxLNaModel:
        iLNb_initializer = SimpleInitializer(iLNb=2)
        iLNa_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNa, 'excitatory', np.linspace(.5,1.5,n_weights))
        r_max_iterator = ParameterIterator(np.linspace(17,20,n_weights), lambda model, r_max: model.set_r_max(Neuron.iLNa, r_max))
        results = Model.compute_states_on_grid([iLNb_initializer], [iLNa_iterator, r_max_iterator])
        results.plot_ratio_diagram()
        plt.xlabel('affinity of input signal to iLNa (AU)')
        plt.ylabel(r'$r_{max}$ for iLNa (AU)')
    plt.savefig(os.path.join('figures/ratios_phase_diagram', folder, f'ratios_phase_diagram_{suffix}.pdf'))
    plt.savefig(os.path.join('figures/ratios_phase_diagram', folder, f'ratios_phase_diagram_{suffix}.png'))
    # plt.show()

def thresholds_combined_model():
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
            go.Scatter3d(x=[20,18],
                     y=[0,10],
                     z=[th([20,0]).item(), th([18,10]).item()],
                     marker={'size':4, 'color':'red'},
                     mode='markers+text',
                     text=['fed', 'sucrose'],
                     textfont_size=18,
                     showlegend=False)
    ]
    layout = {
      'scene': {
            'camera':{
                  'eye':{'x':.39, 'y':-2.47, 'z':.65}
                  },
            'xaxis_title': 'rmax(iLNa)',
            'yaxis_title': 'i(Hb)',
            'zaxis_title': 'w(Mch->iLNa)',
            },
            'font' : {'size':12},
            'autosize':False,
            'width':700,
            'height':700
    }
    fig = go.Figure(data=data, layout=layout)
    fig.write_image('figures/thresholds/fed_and_sucrose_threshold_plotly.pdf')
    fig.write_image('figures/thresholds/fed_and_sucrose_threshold_plotly.png')
    # fig.show()

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
            go.Scatter3d(x=[20,18],
                     y=[0,10],
                     z=[th([20,0]).item(), th([18,10]).item()],
                     marker={'size':4, 'color':'red'},
                     mode='markers',
                     showlegend=False)
    ]
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
    fig.write_image('figures/thresholds/fed_and_sucrose_threshold_plotly_no_legend.pdf')
    fig.write_image('figures/thresholds/fed_and_sucrose_threshold_plotly_no_legend.png')


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    # neuron_activity(InputCurrentHbModel, 'model_input_current_to_handleB', 'with_input_current_to_handleB')
    # neuron_activity(DecreasedRmaxLNaModel, 'model_decreased_rmax_to_iLNa', 'with_decreased_rmax_to_iLNa')
    # neuron_activity(CombinedModel, 'combined_model', 'combined_model')

    # phase_diagram(InputCurrentHbModel, 'model_input_current_to_handleB', 'with_input_current_to_handleB')
    # phase_diagram(DecreasedRmaxLNaModel, 'model_decreased_rmax_to_iLNa', 'with_decreased_rmax_to_iLNa')
    thresholds_combined_model()
    os.system('''cp -R "figures/." "/home/alexandre/Insync/blanc.alexandre.perso@gmail.com/Google Drive - Shared with me/For_submission/Figures/plots_from_alexandre/final"''')