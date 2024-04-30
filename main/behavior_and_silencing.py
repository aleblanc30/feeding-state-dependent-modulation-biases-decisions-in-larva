from models import *
import matplotlib.pyplot as plt
import itertools
import os

def main():
    Model = CombinedModel

    n_weights = 100

    iLNa_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNa, "excitatory", np.linspace(.5,1.5,n_weights))
    iLNb_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNb, "excitatory", np.linspace(1.5,2.5,n_weights))
    silencingHb_iterator = SilencingIterator(Neuron.fbLNHb)
    internalStateIterator = Model.state_iterator

    results = Model.compute_states_on_grid([], [internalStateIterator, silencingHb_iterator, iLNa_iterator, iLNb_iterator])



    _, axs = plt.subplots(2, 2, figsize=(16,16))
    plt.suptitle('HandleB silencing experiments.')
    for ax, (condition, silencing) in zip(axs.ravel(), itertools.product(('fed', 'sucrose'), (True, False))):
        results.plot_partial_ratio_diagram([iLNa_iterator, iLNb_iterator], (condition, silencing), ax)
        ax.set_xlabel('iLNa')
        ax.set_ylabel('iLNb')

        ax.set_title(condition + ', HandleB ' + ("" if silencing else "not ") + "silenced")
    plt.savefig('figures/silencing/combined_model/ratios_phase_diagram_modification_under_HandleB_silencing_combined_model.pdf')
    plt.savefig('figures/silencing/combined_model/ratios_phase_diagram_modification_under_HandleB_silencing_combined_model.pdf')
    # plt.show()
    ######################################################

    n_weights = 100

    iLNa_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNa, "excitatory", np.linspace(.5,1.5,n_weights))
    iLNb_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNb, "excitatory", np.linspace(1.5,2.5,n_weights))
    silencingiLNa_iterator = SilencingIterator(Neuron.iLNa)
    internalStateIterator = Model.state_iterator
    results = Model.compute_states_on_grid([], [internalStateIterator, silencingiLNa_iterator, iLNa_iterator, iLNb_iterator])


    _, axs = plt.subplots(2, 2, figsize=(16,16))
    plt.suptitle('iLNa silencing experiments.')
    for ax, (condition, silencing) in zip(axs.ravel(), itertools.product(('fed', 'sucrose'), (True, False))):
        results.plot_partial_ratio_diagram([iLNa_iterator, iLNb_iterator], (condition, silencing), ax)
        ax.set_xlabel('iLNa')
        ax.set_ylabel('iLNb')

        ax.set_title(condition + ', iLNa ' + ("" if silencing else "not ") + "silenced")
    plt.savefig('figures/silencing/combined_model/ratios_phase_diagram_modification_under_iLNa_silencing_combined_model.pdf')
    plt.savefig('figures/silencing/combined_model/ratios_phase_diagram_modification_under_iLNa_silencing_combined_model.png')
    # plt.show()
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    main()