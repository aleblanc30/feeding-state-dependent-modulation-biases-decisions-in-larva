from models import *
import os
import matplotlib.pyplot as plt

def main():
    Model = BaseModel

    n_weights = 100

    iLNa_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNa, "excitatory", np.linspace(.5,1.5,n_weights))
    iLNb_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNb, "excitatory", np.linspace(1.5,2.5,n_weights))
    silencingHb_iterator = SilencingIterator(Neuron.fbLNHb)

    results = Model.compute_states_on_grid([], [silencingHb_iterator, iLNa_iterator, iLNb_iterator])




    _, axs = plt.subplots(1, 2, figsize=(20,8))
    plt.suptitle('Hb silencing blunts action selection')
    for ax, silencing in zip(axs, (True, False)):
        results.plot_partial_ratio_diagram([iLNa_iterator, iLNb_iterator], (silencing,), ax)
        ax.set_xlabel('iLNa')
        ax.set_ylabel('iLNb')
        ax.set_title('Hb silenced' if silencing else 'Hb functional')

    plt.savefig('figures/supplementary/Hb_inhibition_sharpens_action_selection.pdf')
    plt.savefig('figures/supplementary/Hb_inhibition_sharpens_action_selection.png')

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    main()