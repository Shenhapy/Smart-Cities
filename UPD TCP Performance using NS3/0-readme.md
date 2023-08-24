# Simulation Results - NS3 Network Analysis

This repository contains the simulation results and analysis for a CSMA network using NS3 software. The purpose of this project is to investigate and compare the performance of TCP and UDP in this specific network setup.

## Prerequisites

Before running the simulations and generating the plots, make sure you have the following installed on your system:

1. NS3.35: The NS3 network simulator is required to run the simulations. Ensure you have the appropriate version (3.35) installed on your system.

2. Gnuplot: Gnuplot is used to plot the results from the simulation data. Make sure you have Gnuplot installed to visualize the performance metrics.

## Running Individual Simulations

To run the simulations for TCP and UDP separately, follow these steps:

1. Copy the files `1-tcp.cpp` and `1-udp.cpp` from this repository.

2. Navigate to the NS3 directory (`ns-allinone-3.35`) in your terminal.

3. To run the TCP simulation, execute the following command:

./waf --run 1-tcp


4. To run the UDP simulation, execute the following command:

./waf --run 1-udp


## Running Multiple Simulations

If you want to perform simulations for multiple files and generate consolidated results, we provide shell scripts `2-TCPall` and `2-UDPall` for this purpose.

1. Copy the files `2-TCPall` and `2-UDPall` to the `ns-allinone-3.35` directory.

2. Open your terminal in the `ns-allinone-3.35` directory.

3. To run all UDP simulations together, execute the following command:

./2-UDPall


4. To run all TCP simulations together, execute the following command:

./2-TCPall


## Plotting Results

To visualize the simulation results and generate plots, we provide a Jupyter notebook named `0-The Plots.jpynb`. Ensure you have Python 3 installed on your system. You can run the notebook locally on your machine or upload it to Google Colab and execute the cells to view all graphs included in the project report.

Additionally, some plots are extracted from the C++ files using Gnuplot. To generate these plots, follow these steps:

1. Navigate to the directory where the files `Flow_vs_Throughputtcp.plt` and `Flow_vs_Throughputudp.plt` exist.

2. Open your terminal in this directory.

3. To generate the TCP flow vs. throughput plot, execute the following command:

gnuplot Flow_vs_Throughputtcp.plt


4. To generate the UDP flow vs. throughput plot, execute the following command:

gnuplot Flow_vs_Throughputudp.plt


These instructions will enable you to replicate the simulation results and generate plots to analyze the performance of TCP and UDP in the specified CSMA network using NS3.
