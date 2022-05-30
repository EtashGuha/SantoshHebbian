from tkinter import N
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np

def draw(name, data):
    with open(data, "rb") as f:
        rounds_to_all= pickle.load(f)
    new_dict = {}



    rounds_to_all[99]
    plot_tau = False
    if not plot_tau:
        for key in rounds_to_all.keys():
            all_values = rounds_to_all[key]
            taus, vals, densities, comps, compstwo, greedy_results, min_degree_results = list(zip(*all_values))
            for i in range(len(vals)):
                if vals[i] not in new_dict:
                    new_dict[vals[i]] = [(key, taus[i], densities[i], comps[i], greedy_results[i], min_degree_results[i])]
                else:
                    new_dict[vals[i]].append((key, taus[i], densities[i], comps[i], greedy_results[i], min_degree_results[i]))
    else:
        for key in rounds_to_all.keys():
            all_values = rounds_to_all[key]
            taus, vals, densities, comps, greedy_results, min_degree_results = list(zip(*all_values))
            for i in range(len(vals)):
                if taus[i] not in new_dict:
                    new_dict[taus[i]] = [(key, vals[i], densities[i], comps[i], greedy_results[i], min_degree_results[i])]
                else:
                    new_dict[taus[i]].append((key, vals[i], densities[i], comps[i], greedy_results[i], min_degree_results[i]))
    import matplotlib.pyplot as plt 
    import math
    counter = 0
    already_plotted_greedy = False
    new_array = []
    for key in new_dict.keys():
        all_values = new_dict[key]
        vals, other_measures, densities, comps, greedy_results, min_degree_results = list(zip(*all_values))
        densities = np.asarray(densities)
        greedies = np.asarray(greedy_results)
        min_degrees = np.asarray(min_degree_results)
        comps = np.asarray(comps)





        idx_of_max = np.argmax(densities)

        new_array.append((key, densities[idx_of_max]))
        plot_comps = False
        if not plot_comps:
            
            plt.plot(vals, densities, label=math.ceil(key * 100)/100)
            if not already_plotted_greedy:
                already_plotted_greedy = True
                ax = plt.axes()
                ax.set(facecolor = "white")
                plt.plot(vals, greedies, label="greedy")
                plt.plot(vals, min_degrees, label="min_degree")
        else:
            print(comps)
            ax = plt.axes()
            ax.set(facecolor = "white")
            print(key)
            plt.plot(vals, comps, label=math.ceil(key * 100)/100)

        plt.title("Effects of Plasticity Parameters on Density over Time")

        plt.xlabel("Round Index")
        plt.ylabel("Density")
        
        plt.legend(title="Plasticity Params")
        plt.savefig(name)

def draw_period_converge(name, data):
    with open(data, "rb") as f:
        rounds_to_all= pickle.load(f)
    
    params = []
    periods = []
    for ele in rounds_to_all[99]:
        params.append(ele[1])
        periods.append(ele[4])
    
    plt.plot(params, periods)
    plt.xlabel("Hebbian Plasticity Parameters")
    plt.ylabel("Percent of Runs Exhibiting 2 Period Oscillations")
    plt.title("Plasticity vs. Probability of 2 Period Oscillation")
    plt.savefig("/nethome/eguha3/SantoshHebbian/plasticity_vs_osci.png")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--name")


    args = parser.parse_args()
    draw_period_converge(args.name, args.data)