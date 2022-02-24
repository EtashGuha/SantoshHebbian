import matplotlib.pyplot as plt
import argparse
import pickle

def draw(name, data):
    with open(data, "rb") as f:
        data = pickle.load(f)

    vals, densities = list(zip(*data))

    plt.scatter(vals, densities)

    plt.xlabel("Plasticity Parameters")
    plt.ylabel("Density")
    plt.savefig(name)
    print('done')

if __name__ == "__main__":
    breakpoint()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--name")


    args = parser.parse_args()
    draw(args.name, args.data)