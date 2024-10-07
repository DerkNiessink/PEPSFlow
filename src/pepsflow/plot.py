import argparse
import matplotlib.pyplot as plt
import os

from pepsflow.train.iPEPS_reader import iPEPSReader


def main():

    parser = argparse.ArgumentParser(prog="Plot iPEPS observables.")
    parser.add_argument("folder", help="Folder containing the data.")
    folder = os.path.join("data", vars(parser.parse_args())["folder"])

    reader = iPEPSReader(folder)
    lambda_values = reader.get_lambdas()

    plt.figure(figsize=(6, 4))
    plt.plot(lambda_values, reader.get_energies(), "v-", markersize=4, linewidth=0.5)
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$E$")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(
        lambda_values, reader.get_magnetizations(), "v-", markersize=4, linewidth=0.5
    )
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\langle M_z \rangle$")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(
        lambda_values, reader.get_correlations(), "v-", markersize=4, linewidth=0.5
    )
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\xi$")
    plt.show()
