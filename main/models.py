import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import itertools
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

RNG = np.random.default_rng(seed=0)

class Neuron:
    Mch    = 0 
    B1     = 1
    B2     = 2  
    iLNb   = 3 
    iLNa   = 4 
    fbLNHa = 5  
    fbLNHb = 6

N = 7

class ParameterIterator:
    def __init__(self, iterator, callback):
        self.iterator = iterator
        self.callback = callback
    
    def __call__(self, model, param):
        return self.callback(model, param)
    
    def __iter__(self):
        return iter(self.iterator)
    
    def __getitem__(self, idx):
        return self.iterator[idx]
    
    def __len__(self):
        return len(self.iterator)
    
class SynapseIterator(ParameterIterator):
    def __init__(self, preNeuron, postNeuron, type, iterator):

        def callback(model, param):
                if type == 'excitatory':
                    model.Aex[postNeuron, preNeuron] = param
                    if preNeuron == Neuron.Mch:
                        if postNeuron == Neuron.iLNa:
                            model.iLNa = param
                        if postNeuron == Neuron.iLNb:
                            model.iLNb = param
                elif type == 'inhibitory':
                    model.Ain[postNeuron, preNeuron] = param
                model.update()

        super().__init__(iterator, callback)

class SilencingIterator(ParameterIterator):
    def __init__(self, neuron):
        def callback(model, flag):
            if flag:
                model.silence(neuron)
            else:
                model.reset_silencing()
        super().__init__([True, False], callback)

class ModelInitializer:
    def __init__(self, initializer):
        self.initializer = initializer
    def __call__(self, model):
        return self.initializer(model)
    
class SimpleInitializer(ModelInitializer):
    def __init__(self, iLNa=None, iLNb=None, alpha=None):
        def initializer(model):
            if iLNa is not None:
                model.iLNa = iLNa
            if iLNb is not None:
                model.iLNb = iLNb
            if alpha is not None:
                model.set_sensitivity(alpha)
            model.reset_silencing()
        super().__init__(initializer)

class ComputeOnGridResults:
    def __init__(self, results, iterators, t_max):
        self.results = results
        self.iterators = iterators
        self.t_max = t_max
        self.shape = tuple(len(it) for it in self.iterators)

    def __getitem__(self, idx):
        return self.results[idx]

    def plot_behavior_diagram(self, classifier, ax=None):
        if ax is None:
            plt.figure(figsize=(7,4))
            ax = plt.gca()
        end_states = np.array([self.results[t].y[[Neuron.B1, Neuron.B2], -1] for t in itertools.product(*self.iterators)])
        behavior = classifier.predict(end_states.reshape(-1,2))
        behavior = behavior.reshape(self.shape)
        behavior = behavior.transpose()
        x, y = self.iterators

        ax.imshow(behavior, extent=[x[0],x[-1],y[0],y[-1]],
                   origin='lower', aspect='auto', cmap=ListedColormap(['b', 'r']))
        
        ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='b', markersize=11),
                           Line2D([0],[0],marker='o',color='w',markerfacecolor='r', markersize=11)],
                  labels=['bend', 'hunch'], loc='upper right')
        
        ax.set_yticks([y[0], (y[0]+y[-1])/2, y[-1]])
        ax.set_xticks([x[0], (x[0]+x[-1])/2, x[-1]])

    def plot_partial_behavior_diagram(self, classifier, iterators, prefix, ax=None):
        if ax is None:
            plt.figure(figsize=(7,4))
            ax = plt.gca()
        end_states = np.array([self.results[prefix+t].y[[Neuron.B1, Neuron.B2], -1] for t in itertools.product(*iterators)])
        behavior = classifier.predict(end_states.reshape(-1,2))
        behavior = behavior.reshape(tuple(len(it) for it in iterators))
        behavior = behavior.transpose()
        x, y = iterators

        ax.imshow(behavior, extent=[x[0],x[-1],y[0],y[-1]],
                   origin='lower', aspect='auto', cmap=ListedColormap(['b', 'r']))
        
        ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='b', markersize=11),
                           Line2D([0],[0],marker='o',color='w',markerfacecolor='r', markersize=11)],
                  labels=['bend', 'hunch'], loc='upper right')
        
        ax.set_yticks([y[0], (y[0]+y[-1])/2, y[-1]])
        ax.set_xticks([x[0], (x[0]+x[-1])/2, x[-1]])

    def plot_ratio_diagram(self, ax=None):
        if ax is None:
            plt.figure(figsize=(7,4))
            ax = plt.gca()
        end_states = np.array([self.results[t].y[[Neuron.B1, Neuron.B2], -1] for t in itertools.product(*self.iterators)])
        end_states = end_states.reshape(-1,2)
        ratios = end_states[:,1]/end_states[:, 0]       
        ratios = ratios.reshape(self.shape)
        ratios = ratios.transpose()
        x, y = self.iterators

        im = ax.imshow(ratios, extent=[x[0],x[-1],y[0],y[-1]],
                   origin='lower', aspect='auto', cmap='RdBu')
        plt.colorbar(im, ax=ax).set_label('B2 activity/B1 activity')
    
        ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='b', markersize=11),
                            Line2D([0],[0],marker='o',color='w',markerfacecolor='r', markersize=11)],
                   labels=['bend', 'hunch'], loc='upper right')
        
        ax.set_xticks([x[0], (x[0]+x[-1])/2, x[-1]])
        ax.set_yticks([y[0], (y[0]+y[-1])/2, y[-1]])
        
    def plot_partial_ratio_diagram(self, iterators, prefix, ax=None):
        if ax is None:
            plt.figure(figsize=(7,4))
            ax = plt.gca()
        end_states = np.array([self.results[prefix+t].y[[Neuron.B1, Neuron.B2], -1] for t in itertools.product(*iterators)])
        end_states = end_states.reshape(-1,2)
        ratios = end_states[:,1]/end_states[:, 0]       
        ratios = ratios.reshape(tuple(len(it) for it in iterators))
        ratios = ratios.transpose()
        x, y = iterators

        im = ax.imshow(ratios, extent=[x[0],x[-1],y[0],y[-1]],
                   origin='lower', aspect='auto', cmap='RdBu', vmin=0, vmax=0.75)
        plt.colorbar(im, ax=ax).set_label('B2 activity/B1 activity')
    
        ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='b', markersize=11),
                            Line2D([0],[0],marker='o',color='w',markerfacecolor='r', markersize=11)],
                   labels=['bend', 'hunch'], loc='upper right')
        
        ax.set_xticks([x[0], (x[0]+x[-1])/2, x[-1]])
        ax.set_yticks([y[0], (y[0]+y[-1])/2, y[-1]])

    def plot_output_diagram(self, function, ax=None, cmap='inferno', label=''):
        if ax is None:
            plt.figure(figsize=(7,4))
            ax = plt.gca()
        values = np.array([function(self[p]) for p in itertools.product(*self.iterators)])
        values = values.reshape(self.shape)
        values = values.transpose()
        x, y = self.iterators

        im = ax.imshow(values, extent=[x[0],x[-1],y[0],y[-1]],
                   origin='lower', aspect='auto', cmap=cmap)
        plt.colorbar(im, ax=ax).set_label(label)
    
        ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='b', markersize=11),
                            Line2D([0],[0],marker='o',color='w',markerfacecolor='r', markersize=11)],
                   labels=['bend', 'hunch'], loc='upper right')
        
        ax.set_xticks([x[0], (x[0]+x[-1])/2, x[-1]])
        ax.set_yticks([y[0], (y[0]+y[-1])/2, y[-1]])
    
    def plot_silencing_curves(self, neuron, xlabel='', ylabel='', prefix='', merge_silenced=False, max_time=None):
        baseIterator = self.iterators[0]
        cmap = colormaps['plasma']
        norm = Normalize(vmin=min(baseIterator), vmax=max(baseIterator))

        fig = plt.figure(figsize=(12,6))
        silenced_plotted = False
        max_simulation_time = 0
        for p in baseIterator:
            silenced_sol, control_sol = self[(p, True)], self[(p, False)]
            max_simulation_time = max(max_simulation_time, max(silenced_sol.t[-1], control_sol.t[-1]))
        max_time = min(max_time, max_simulation_time) if max_time is not None else max_simulation_time
        for p in baseIterator:
            silenced_sol, control_sol = self[(p, True)], self[(p, False)]
            
            silenced_y, silenced_t = silenced_sol.y[neuron,silenced_sol.t <= max_time], silenced_sol.t[silenced_sol.t <= max_time]
            extend_silenced_t = np.linspace(silenced_t[-1], max_time, 11)[1:]
            silenced_y, silenced_t = np.concatenate((silenced_y, silenced_y[-1]*np.ones_like(extend_silenced_t))), np.concatenate((silenced_t, extend_silenced_t))
            
            control_y, control_t = control_sol.y[neuron, control_sol.t <= max_time], control_sol.t[control_sol.t <= max_time]
            extend_control_t = np.linspace(control_t[-1], max_time, 11)[1:]
            control_y, control_t = np.concatenate((control_y, control_y[-1]*np.ones_like(extend_control_t))), np.concatenate((control_t, extend_control_t))

            if not(silenced_plotted):
                if merge_silenced:
                    silenced_plotted = True
                    plt.plot(silenced_t, silenced_y, '-', color='k', lw=6)
                else:
                    plt.plot(silenced_t, silenced_y, '-', color=cmap(norm(p)), lw=4)
            plt.plot(control_t, control_y, '--', color=cmap(norm(p)), lw=4)
        plt.legend(labels=('silenced', 'not silenced'), handles=[Line2D([],[],ls='-', c='k', lw=4), Line2D([],[],ls='--', c='k', lw=4)], loc='upper right', handlelength=4)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar(ScalarMappable(norm, cmap), label='w_iLNa', location='right')
        return fig
    



class BaseModel:
    state_iterator = ParameterIterator(["fed", "sucrose"], lambda m, p : m.set_state(p))

    def __init__(self, iLNa=0, iLNb=0, alpha=.5):
        self.iLNa, self.iLNb = iLNa, iLNb
        self.sensitivity = self.sensitivity_factory(alpha)
        self.Aex = self.Aex_factory(iLNa, iLNb)
        self.Ain = self.Ain_factory()
        self.V0 = np.array([0, 20, 20, 20, 20, 20, 20])
        self.exogenous_current = np.zeros(7)
        self.tau = np.array([1., 35., 35., 35., 35., 35., 35.])
        self.r_max = 20*np.ones(N)
        self.stimulus = self.stimulus_factory()
        self.N = 7
        self.k_ex = 2.5
        self.non_negative = True

        self.dynamics, self.input, self.jac = self.dynamics_factory()
        self.steady_state_event = self.steady_state_event_factory()

        self.previous_event_value = 1e3

    def set_Aex(self, *args, **kwargs):
        if args:
            assert(len(args) == 1)
            self.Aex = args[0]
            self.update()
        else:
            self.Aex = self.Aex_factory(**kwargs)
            self.update()

    def set_Ain(self, *args, **kwargs):
        if args:
            assert(len(args) == 1)
            self.Ain = args[0]
            self.update()
        else:
            self.Ain = self.Ain_factory(**kwargs)
            self.update()

    def set_synapse(self, presynapticNeuron, postsynapticNeuron, coupling):
        if coupling > 0:
            self.Ain[presynapticNeuron, postsynapticNeuron] = 0
            self.Aex[presynapticNeuron, postsynapticNeuron] = coupling
        else:
            self.Ain[presynapticNeuron, postsynapticNeuron] = -coupling
            self.Aex[presynapticNeuron, postsynapticNeuron] = 0
        self.update()

    def set_r_max(self, neuron, rmax):
        self.r_max[neuron] = rmax
        self.update()

    def set_exogenous_current(self, neuron, current):
        self.exogenous_current[neuron] = current
        self.update()

    def set(self, **kwargs):
        attrs = vars(self)
        for k, v in kwargs.items():
            if not(k in attrs):
                raise ValueError(f'No {k} parameter.')
            attrs[k] = v
        self.update()

    def silence(self, silenced):
        self.Ain[:, silenced] = 0
        self.Aex[:, silenced] = 0
        self.update()

    def reset_silencing(self):
        self.Aex = self.Aex_factory(self.iLNa, self.iLNb)
        self.Ain = self.Ain_factory()
        self.update()

    def run(self, init, t_max):
        self.update()
        t = np.linspace(0, t_max, 500)
        try:
            sol = solve_ivp(self.dynamics, (0, t_max), init, t_eval=t, events=self.steady_state_event, method='LSODA', rtol=1e-3, atol=1e-3)
        except ValueError:
            sol = solve_ivp(self.dynamics, (0, t_max), init, t_eval=t, method='LSODA', rtol=1e-3, atol=1e-3)

        self.sol = sol
        self.states = self.sol.y
        return sol

    def update(self):
        self.dynamics, self.input, self.jac = self.dynamics_factory()
        self.steady_state_event = self.steady_state_event_factory()

    def Aex_factory(self, w_iLNa, w_iLNb, w_Hb=None, silenced=[]):
        Aex = np.array([[     0,  0,  0, 0, 0, 0, 0],
                        [   1.5,  0,  0, 0, 0, 0, 0],
                        [   .75,  0,  0, 0, 0, 0, 0],
                        [w_iLNb,  0,  0, 0, 0, 0, 0],
                        [w_iLNa,  0,  0, 0, 0, 0, 0],
                        [     0, .2, .2, 0, 0, 0, 0],
                        [    .4,  0, .5, 0, 0, 0, 0]])
        if w_Hb is not None:
            Aex[Neuron.fbLNHb,[Neuron.Mch, Neuron.B2]] = w_Hb
        Aex[:, silenced] = 0
        return Aex

    def Ain_factory(self, silenced=[]):
        Ain = np.array([[0, 0, 0,      0,      0,      0,      0],
                        [0, 0, 0, 1.7648, 1.3841,      0,      0],
                        [0, 0, 0,      1, 5.9167,      0,      0],
                        [0, 0, 0,      0, 3.3744, 1.6659,  2.191],
                        [0, 0, 0, 2.7133,      0, 1.1010, 3.3031],
                        [0, 0, 0, 1.8411, 1.1158,      0,      0],
                        [0, 0, 0, 1.7331, 2.2145,      0,      0]])
        Ain[:, silenced] = 0
        return Ain
    
    def sensitivity_factory(self, alpha):
        sensitivity = np.ones((7,7))
        sensitivity[Neuron.iLNa, Neuron.Mch] = .5*(1-alpha) + 1.5*alpha
        return sensitivity
    
    def set_sensitivity(self, alpha):
        self.sensitivity = self.sensitivity_factory(alpha)
        self.update()

    def set_state(self, state):
        assert state in ['fed', 'sucrose']

    def stimulus_factory(self, start_time=0, stimulus_length=450, amplitude=2):
        def s(t):
            s_ = np.zeros(N)
            if start_time <= t and t < start_time+stimulus_length:
                s_[0] = amplitude
            return s_
        return s

    def dynamics_factory(self):
        def input(t, r):
            return self.k_ex*(self.r_max-r)*(self.Aex.dot(r)) - self.Ain.dot(r)

        def dynamics(t, r):
            return 1/self.tau * (-self.V0 - r + self.stimulus(t) + self.exogenous_current + self.k_ex*(self.r_max-r)*((self.sensitivity*self.Aex).dot(r)) - self.Ain.dot(r))

        dynamics = BaseModel.non_negative(dynamics) if self.non_negative else dynamics

        def jac(t, r):
            return 1/self.tau.reshape(self.N,1) * (-np.eye(self.N) + self.k_ex*(self.r_max-r).reshape(self.N,1)*self.Aex - np.diag(self.Aex.dot(r)) - self.Ain)
            
        return dynamics, input, jac

    def non_negative(dynamics):
        def non_negative_dynamics(t, r):
            r = np.maximum(r, 0)
            dr = dynamics(t, r)
            dr = np.where(r <= 1e-9, np.maximum(dr, 0), dr)
            return dr
        return non_negative_dynamics

    def steady_state_event_factory(self, eps=1e-3):
        def steady_state_reached(t, r):
            dr = self.dynamics(t, r)
            max_dr = np.max(np.abs(dr))
 

            return max_dr-eps
        steady_state_reached.terminal = True
        steady_state_reached.direction = -1
        return steady_state_reached
    
    def run_with_noise(self, init, t_max, sigma=0):
        self.non_negative = False
        self.update()
        t_eval = np.linspace(0, t_max, 500)
        class sol:
            pass

        eps = 1e-3
        n_steps = int(t_max/eps)+1
        t_arr = eps * np.arange(n_steps+1)
        states = np.zeros((N, len(t_arr)))
        for i in range(n_steps-1):
            r = np.maximum(states[:,i], 0)
            dr = eps*self.dynamics(t_arr[i], states[:,i]) + sigma*np.sqrt(eps)*RNG.normal(size=N)*(self.Aex[:,Neuron.Mch]-self.Ain[:,Neuron.Mch])
            dr = np.where(r <= 1e-9, np.maximum(dr, 0), dr)
            states[:,i+1] = states[:,i] + dr

        y = interp1d(t_arr, states, axis=-1)
        sol.t = t_eval
        sol.y = y(t_eval)

        self.sol = sol
        self.states = self.sol.y
        self.non_negative = True
        return self.sol
    
    @classmethod
    def compute_states_on_grid(cls, initializers, iterators, final_callback=None, t_max=450):
        results = {}
        model = cls()
        for init in initializers:
            init(model)
        for params in itertools.product(*iterators):
            for p, iterator in zip(params, iterators):
                iterator(model, p)
            initial_state = np.zeros(N)
            sol = model.run(initial_state, t_max)
            if final_callback is not None:
                result = final_callback(sol)
            else:
                result = sol
            results[params] = result
        return ComputeOnGridResults(results, iterators, t_max)

    @classmethod
    def compute_thresholds_on_grid(cls, initializers, iterators, output_thresh=.3, t_max=450, bracket=[.5,1.5]):
        results = {}
        model = cls()
        for init in initializers:
            init(model)
        for params in itertools.product(*iterators):
            for p, iterator in zip(params, iterators):
                iterator(model, p)
            initial_state = np.zeros(N)
            def f(iLNa):
                model.Aex[Neuron.iLNa,Neuron.Mch] = iLNa
                model.iLNa = iLNa
                model.update()
                sol = model.run(initial_state, t_max)
                output = sol.y[Neuron.B1,-1]/sol.y[Neuron.B2,-1]-output_thresh
                return output
            try:
                results[params] = root_scalar(f, method='bisect', bracket=bracket, maxiter=10).root
            except:
                results[params] = None
        return ComputeOnGridResults(results, iterators, t_max)
    
class InputCurrentHbModel(BaseModel):
    def set_state(self, state):
        assert state in ['fed', 'sucrose']
        self.set_exogenous_current(Neuron.fbLNHb, 0 if state == "fed" else 10)

class DecreasedRmaxLNaModel(BaseModel):
    def set_state(self, state):
        assert state in ['fed', 'sucrose']
        self.set_r_max(Neuron.iLNa, 20 if state == "fed" else 18)

class CombinedModel(BaseModel):
    def set_state(self, state):
        assert state in ['fed', 'sucrose']
        self.set_r_max(Neuron.iLNa, 20 if state == "fed" else 18)
        self.set_exogenous_current(Neuron.fbLNHb, 0 if state == "fed" else 10)

class AnqiModel(BaseModel):
    def __init__(self, iLNa=0, iLNb=0):
        super().__init__(iLNa, iLNb)
        self.Aex = self.Aex_factory(iLNa, iLNb)
        self.Ain = self.Ain_factory()
        self.V0 = np.array([0, 20, 30, 20, 20, 20, 60])
        self.tau = np.array([1., 20., 20., 20., 20., 20., 20.])
        self.r_max = 40
        self.stimulus = self.stimulus_factory(amplitude=5)
        self.N = 7
        self.k_ex = 1.0
        self.non_negative = True

        self.dynamics, self.input, self.jac = self.dynamics_factory()
        self.steady_state_event = self.steady_state_event_factory()

        self.previous_event_value = 1e3

    def stimulus_factory(self, start_time=0, stimulus_length=450, amplitude=2):
        return super().stimulus_factory(start_time=start_time, stimulus_length=stimulus_length, amplitude=amplitude)

class ModifiedCaseyModel(BaseModel):
    def dynamics_factory(self):
        def input(t, r):
            return (self.r_max-r)*(self.k_ex*self.Aex.dot(r) - self.Ain.dot(r))

        def dynamics(t, r):
            return 1/self.tau * (-self.V0 - r + self.stimulus(t) + (self.r_max-r)*(self.k_ex*self.Aex.dot(r) - self.Ain.dot(r)))

        dynamics = ModifiedCaseyModel.non_negative(dynamics) if self.non_negative else dynamics

        def jac(t, r):
            return 1/self.tau.reshape(self.N,1) * (-np.eye(self.N) + (self.r_max-r).reshape(self.N,1)*(self.k_ex*self.Aex - self.Ain) - np.diag((self.k_ex*self.Aex - self.Ain).dot(r)))

        return dynamics, input, jac

class KimModel(BaseModel):
    def __init__(self, iLNa=0, iLNb=0, g=1, th=.1, k_ex=1.5):
        super().__init__(iLNa, iLNb)
        self.gain = np.array([1, g, g, g, g, g, g])
        self.activation = scipy.special.expit
        self.V0 = np.array([0, th, th, th, th, th, th])
        self.k_ex = k_ex

    def dynamics_factory(self):
        def input(t, r):
            return self.activation((self.k_ex*self.Aex-self.Ain).dot(r) - self.V0)

        def dynamics(t, r):
            return 1/self.tau * (-r + self.stimulus(t) + self.gain*self.activation((self.k_ex*self.Aex-self.Ain).dot(r)-self.V0 ))

        dynamics = KimModel.non_negative(dynamics) if self.non_negative else dynamics

        def jac(t, r):
            raise NotImplementedError()

        return dynamics, input, jac
