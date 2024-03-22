from models import *
import os


def main1():
    V0 = 11
    tau = 1.
    r_max = 4
    k_ex = 2.5
    tmax = 20
    def dynamics(t,r):
        dr = 1/tau*(V0 - r + k_ex*(r - r_max)*r)
        return np.where(r >= 0, dr, np.maximum(dr, 0))

    Delta = (r_max+1/k_ex)**2-4*V0/k_ex
    r_eq1 = .5*(r_max+1/k_ex - np.sqrt(Delta))
    r_eq2 = .5*(r_max+1/k_ex + np.sqrt(Delta))


    sol1 = solve_ivp(dynamics, (0,tmax), [0], t_eval=np.linspace(0,tmax,1000))
    print(sol1.success)
    sol2 = solve_ivp(dynamics, (0,tmax), [r_eq2-0.0001], t_eval=np.linspace(0,tmax,1000))
    print(sol2.success)
    x, y = np.meshgrid(np.linspace(0, tmax, 20), np.linspace(0, 5, 20))
    # x = x.flatten()
    # y = y.flatten()
    v = dynamics(0, y)
    u = np.ones_like(v)
    plt.streamplot(x, y, u, v)
    plt.axhline(r_eq1, 0, tmax, linestyle='--', color='r', lw=4)
    plt.axhline(r_eq2, 0, tmax, linestyle='--', color='r', lw=4)
    plt.plot(sol1.t, sol1.y[0], 'orange', lw=4)
    plt.plot(sol2.t, sol2.y[0], 'orange', lw=4)

    plt.savefig('figures/supplementary/theroretical_show_trajectory.pdf')
    plt.savefig('figures/supplementary/theroretical_show_trajectory.png')

def main2():
    n=5
    _, axs = plt.subplots(1,n, sharex=True, sharey=True, figsize=(15,15/n))
    tau = 3.5
    for i, k_ex in enumerate(np.logspace(-1,1,n)):
        ax = axs[i]
        tmax = 20
        r_m, r_r = 3, 1.5
        V0 = k_ex*r_r*r_m
        r_max = r_m+r_r-1/k_ex
        def dynamics(t,r):
            dr = 1/tau*(V0 - r + k_ex*(r - r_max)*r)
            return np.where(r >= 0, dr, np.maximum(dr, 0))

        Delta = (r_max+1/k_ex)**2-4*V0/k_ex
        r_eq1 = .5*(r_max+1/k_ex - np.sqrt(Delta))
        r_eq2 = .5*(r_max+1/k_ex + np.sqrt(Delta))


        sol = solve_ivp(dynamics, (0,tmax), [0], t_eval=np.linspace(0,tmax,1000))
        x, y = np.meshgrid(np.linspace(0, tmax, 10), np.linspace(0, 5, 30))
        v = dynamics(0, y)
        u = np.ones_like(v)
        ax.streamplot(x, y, u, v)
        ax.plot(sol.t, sol.y[0], 'r', lw=3)
        ax.axhline(r_eq1, 0, tmax, linestyle='--', color='r')
        ax.axhline(r_eq2, 0, tmax, linestyle='--', color='r')

        ax.set_title("$k_{ex} = "+f"{k_ex:.2f}$")

    plt.savefig('figures/supplementary/theoretical_k_ex_is_time_constant.pdf')
    plt.savefig('figures/supplementary/theoretical_k_ex_is_time_constant.png')


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    main1()
    main2()
    os.system('''cp -R "figures/." "/home/alexandre/Insync/blanc.alexandre.perso@gmail.com/Google Drive - Shared with me/For_submission/Figures/plots_from_alexandre/final"''')