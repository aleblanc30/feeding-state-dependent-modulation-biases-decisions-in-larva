# Model

The differential equation defining the evolution of the system is :

$\tau \dot r(t) = - V_0 - r(t) + s(t) + i + k_{ex}(r(t)-r_{max})\odot(A_{ex}r-A_{in}r)$

The parameters can be interpreted as follows according to *Jovanic et al., 2016*:
- $V_0$ : activation threshold
- $s$ : time varying stimulus
- $i$ : constant stimulus/input
- $k_{ex}$ : sensitivity to overall input
- $A_{in}$, $A_{ex}$ : inhibition and excitation connectivity matrix
- $r$ : rate/voltage/neuron activation

# Hypothesis 1 : Sucrose presence induces an incoming current to fbLNHb which sensibilizes it to excitatory input

Modelling interpretation : $i$ contains a nonnegative term for fbLNHb in the sucrose state

## Reproducing behavior observations

Desired outcomes in presence of sucrose :
- frequency of hunch diminishes
- frequency of bend increases

$\implies$ modelling by varying Hb bias to model presence of sucrose and LNa coupling to model the variability in stimuli. 

$\implies$ The decision boundary should shift to increase the size of the bend domain as Hb input current augments.


## Silencing experiments 

1. silencing fbLNHb in presence of sucrose results in increased activity of LNa 
2. silencing LNA in fed state results in increased activity of Hb
3. silencing LNA in sucrose state results in no change of Hb activity

We model the fed state as having no input current on Hb and the sucrose state as having a 10 AU input current on Hb. We compare the trajectories with and without silencing across the range of iLNa inputs.

# Hypothesis 2 : Sucrose diminishes coupling between Mch and iLNa 

Modelling interpretation : the coupling between Mch and iLNa in $A_{ex}$ is decreased.

This is a bit difficult because we usually vary the value of this synapse to account for stimulus variation.
Then we need to consider a new parameter, which would be a stimulus affinity coefficient $\alpha$. For $\alpha = 1$ the stimulus is biased towards iLNa, and for $\alpha = 0$ the stimulus is biased towards iLNb.

From an implementation point of view, we multiply the excitatory matrix by the following mask ($\sigma$ stands for sensitivity) :

$\sigma = \begin{pmatrix} 
1 & 1 & 1 & 1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 \\
.5(1-\alpha)+1.5\alpha & 1 & 1 & 1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 \\
1 & 1 & 1 & 1 & 1 & 1 & 1 
\end{pmatrix}$

so that the input current to iLNa varies continuously from 1AU to from 3AU as $\alpha$ varies from 0 to 1, since the current input from mechanoCh is 2AU.

We can vary the values of iLNa while independently traversing the input space and simulate the model according to the equation :

$\tau \dot r(t) = - V_0 - r(t) + s(t) + i + k_{ex}(r(t)-r_{max})\odot((\sigma \odot A_{ex})r-A_{in}r)$

# Progress meeting 11/04

**Hypothesis 1** iLNa activity should decrease as input current increases. I need to check for that

Maybe we should only consider synaptic strength that yield low activity on LNa.

Maybe we could try to set HandleB input current to 15 instead of 10.

Did I get the colors backward in the diagram for the second hypothesis ? $\implies$ I got the colors right but messed up the interpretation because of the way $\alpha$ was defined. This is now fixed.

Does decreasing the activity of Gridle2 (=LNa) increase the activity ?

hypothesis 1 : add input current

hypothesis 2 : shift synaptic strength/reduce maximum


Silencing handle B and look at behavior output ?
