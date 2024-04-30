from models import *
import os

def main(Model, folder, suffix):
    n_weights = 5

    iLNb_initializer = SimpleInitializer(iLNb=2)
    sucrose_initializer = ModelInitializer(lambda model: model.set_state('sucrose'))
    iLNa_iterator = SynapseIterator(Neuron.Mch, Neuron.iLNa, 'excitatory', np.linspace(.5,1.5,n_weights))
    silencingiLNa_iterator = SilencingIterator(Neuron.iLNa)
    silencingHb_iterator = SilencingIterator(Neuron.fbLNHb)
    fed_initializer = ModelInitializer(lambda model: model.set_state('fed'))

    ##################
    #
    #   SILENCING iLNa IN SUCROSE STATE
    #
    #############


    results = Model.compute_states_on_grid([iLNb_initializer, sucrose_initializer],
                                           [iLNa_iterator, silencingiLNa_iterator])

    fig = results.plot_silencing_curves(Neuron.fbLNHb, xlabel='time (AU)', ylabel='fbLNHb activity (AU)', prefix='iLNa = ', merge_silenced=True, max_time=200)
    fig.suptitle('Silencing iLNa in sucrose condition')
    plt.savefig(os.path.join('figures/silencing', folder, 'iLNA_silencing_in_sucrose_state_' + suffix + '.pdf'))
    plt.savefig(os.path.join('figures/silencing', folder, 'iLNA_silencing_in_sucrose_state_' + suffix + '.png'))
    plt.show()

    ##################
    #
    #   SILENCING Hb IN PRESENCE OF SUCROSE
    #
    #############

    results = Model.compute_states_on_grid([iLNb_initializer, sucrose_initializer],
                                        [iLNa_iterator, silencingHb_iterator])

    fig = results.plot_silencing_curves(Neuron.iLNa, xlabel='time (AU)', ylabel='iLNa activity (AU)', prefix='iLNa = ', max_time=200)
    fig.suptitle('Silencing Hb in sucrose condition')
    plt.savefig(os.path.join('figures/silencing', folder, 'handleB_silencing_in_sucrose_state_' + suffix + '.pdf'))
    plt.savefig(os.path.join('figures/silencing', folder, 'handleB_silencing_in_sucrose_state_' + suffix + '.png'))
    plt.show()

    ##################
    #
    #   SILENCING iLNa IN FED STATE
    #
    #############

    results = Model.compute_states_on_grid([iLNb_initializer, fed_initializer],
                                        [iLNa_iterator, silencingiLNa_iterator])

    fig = results.plot_silencing_curves(Neuron.fbLNHb, xlabel='time (AU)', ylabel='fbLNHb activity (AU)', prefix='iLNa = ', merge_silenced=True, max_time=200)
    fig.suptitle('Silencing iLNa in fed condition')
    plt.savefig(os.path.join('figures/silencing', folder, 'iLNA_silencing_in_fed_state_' + suffix + '.pdf'))
    plt.savefig(os.path.join('figures/silencing', folder, 'iLNA_silencing_in_fed_state_' + suffix + '.png'))
    plt.show()

    ##################
    #
    #   SILENCING Hb IN FED STATE
    #
    #############

    results = Model.compute_states_on_grid([iLNb_initializer, fed_initializer],
                                        [iLNa_iterator, silencingHb_iterator])

    fig = results.plot_silencing_curves(Neuron.iLNa, xlabel='time (AU)', ylabel='iLNa activity (AU)', prefix='iLNa = ', max_time=200)
    fig.suptitle('Silencing handleB in fed condition')
    plt.savefig(os.path.join('figures/silencing', folder, 'handleB_silencing_in_fed_state_' + suffix + '.pdf'))
    plt.savefig(os.path.join('figures/silencing', folder, 'handleB_silencing_in_fed_state_' + suffix + '.png'))
    plt.show()

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    main(InputCurrentHbModel, 'model_input_current_to_handleB', 'with_input_current_to_handleB')
    main(DecreasedRmaxLNaModel, 'model_decreased_rmax_to_iLNa', 'with_decreased_rmax_to_iLNa')
    main(CombinedModel, 'combined_model', 'combined_model')