# Author: Madelynn Mackenzie
# Contact email: madelynnmack@gmail.com
# Mentor: Christopher S. Dunham
# Email: csdunham@chem.ucla.edu, csdunham@protomail.com
# This is an original work unless otherwise noted.

# Function to perform power-law analysis of cardiomyocyte pacemaker 
# translocations.
import numpy as np
from numpy.lib.histograms import histogram
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import scipy as sp
import statsmodels as sm
import powerlaw as pl
import detect_transloc


def pl_histogram_plotting(analysisGUI, pace_maker, batch_data): 

    # Check whether using translocation data from individual dataset or batch.
    if hasattr(pace_maker, "transloc_events"):
        size_event = pace_maker.transloc_events
        print("Using dataset translocations.")
    elif hasattr(batch_data, "batch_translocs"):
        size_event = batch_data.batch_translocs
        print("Using batch translocations.")
    
    sorted_size_event = sorted(size_event)
    
    #Model Distributions for Comparison
    #Power Law
    pldist = sp.stats.powerlaw
    args_powerlaw = pldist.fit(sorted_size_event)

    #Exponential
    expondist = sp.stats.expon
    args_expon = expondist.fit(sorted_size_event)

    #Stretched Exponential - Weibull
    weibulldist = sp.stats.exponweib
    args_exponweib = weibulldist.fit(sorted_size_event)

    #Log Normal
    lognormdist = sp.stats.lognorm
    args_lognorm = lognormdist.fit(sorted_size_event)

    #Finding the max x value and linerizing the model data
    bmax = max(sorted_size_event)
    beats = np.linspace(0., bmax, 100)

    #Fitting Distributions to Data
    dist_powerlaw = pldist.pdf(beats, *args_powerlaw)
    dist_lognormal = lognormdist.pdf(beats, *args_lognorm)
    dist_exponweib = weibulldist.pdf(beats, *args_exponweib)
    dist_expon = expondist.pdf(beats, *args_expon)

    #Plotting Distributions
    analysisGUI.plWindow.powerlawPlot.axis1.cla()

    #when this is here, the axis is cleared 
    # (takes away the title of the plot as defined in the window)
    analysisGUI.plWindow.powerlawPlot.axis1.set(
        title ="Distribution Fit Comparision", 
        xlabel="Event Size (beats)", 
        ylabel="Number of Events" )

    nbins = 30
    length = len(sorted_size_event)

    analysisGUI.plWindow.powerlawPlot.axis1.hist(sorted_size_event, nbins)
    # plt.title("Number of Events vs Event Size")
    analysisGUI.plWindow.powerlawPlot.axis1.plot(
        beats, 
        dist_lognormal * len(sorted_size_event) * bmax / nbins,
        '--c', 
        lw=3, 
        label='Log Normal')
    analysisGUI.plWindow.powerlawPlot.axis1.plot(
        beats, 
        dist_exponweib * len(sorted_size_event) * bmax / nbins,
        '-g', 
        lw=3, 
        label='Weibull')
    analysisGUI.plWindow.powerlawPlot.axis1.plot(
        beats, 
        dist_expon * len(sorted_size_event) * bmax / nbins,
        '--y', 
        lw=3, 
        label='Exponential')
    analysisGUI.plWindow.powerlawPlot.axis1.plot(
        beats, 
        dist_powerlaw * len(sorted_size_event) * bmax / nbins,
        '-r', 
        lw=3, 
        label='Powerlaw')
    analysisGUI.plWindow.powerlawPlot.axis1.set_xlabel("Event Size (beats)")
    analysisGUI.plWindow.powerlawPlot.axis1.set_ylabel("Number of Events")
    analysisGUI.plWindow.powerlawPlot.axis1.legend()
    analysisGUI.plWindow.powerlawPlot.axis1.set_ylim(0,100)
    analysisGUI.plWindow.powerlawPlot.fig.tight_layout()
    analysisGUI.plWindow.powerlawPlot.draw()

def pl_truncated_histogram_plotting(analysisGUI, pace_maker, batch_data): 

     # Check whether using translocation data from individual dataset or batch.
    if hasattr(pace_maker, "transloc_events"):
        size_event = pace_maker.transloc_events
        print("Using dataset translocations.")
    elif hasattr(batch_data, "batch_translocs"):
        size_event = batch_data.batch_translocs
        print("Using batch translocations.")

    sorted_size_event = sorted(size_event)
    
    #Removing data below x_min
    PL_results = pl.Fit(sorted_size_event)
    truncated_sorted_size_event = [
        x for x in sorted_size_event if x >= PL_results.power_law.xmin]
    
    #Model Distributions for Comparison
    pldist = sp.stats.powerlaw
    args_powerlaw = pldist.fit(truncated_sorted_size_event)

    expondist = sp.stats.expon
    args_expon = expondist.fit(truncated_sorted_size_event)

    weibulldist = sp.stats.exponweib
    args_exponweib = weibulldist.fit(truncated_sorted_size_event)

    lognormdist = sp.stats.lognorm
    args_lognorm = lognormdist.fit(truncated_sorted_size_event)

    #Finding the max x value and linerizing the model data
    bmax = max(truncated_sorted_size_event)
    beats = np.linspace(0., bmax, 100)

    #Fitting Distributions to Data
    dist_powerlaw = pldist.pdf(beats, *args_powerlaw)
    dist_lognormal = lognormdist.pdf(beats, *args_lognorm)
    dist_exponweib = weibulldist.pdf(beats, *args_exponweib)
    dist_expon = expondist.pdf(beats, *args_expon)

    #Plotting Distributions
    analysisGUI.plWindow.powerlawPlot.axis2.cla()

    PL_results = pl.Fit(sorted_size_event)
    x_min = PL_results.power_law.xmin

    analysisGUI.plWindow.powerlawPlot.axis2.set(
        title = "X values below {} removed".format(x_min),
        xlabel="Event Size (beats)", 
        ylabel="Number of Events" )

    nbins = 30
    length = len(sorted_size_event)

    analysisGUI.plWindow.powerlawPlot.axis2.hist(
        truncated_sorted_size_event, 
        nbins)

    analysisGUI.plWindow.powerlawPlot.axis2.plot(
        beats, 
        dist_lognormal * len(truncated_sorted_size_event) * bmax / nbins,
        '--c', 
        lw=3, 
        label='Log Normal')
    analysisGUI.plWindow.powerlawPlot.axis2.plot(
        beats, 
        dist_exponweib * len(truncated_sorted_size_event) * bmax / nbins,
        '-g', 
        lw=3, 
        label='Weibull')
    analysisGUI.plWindow.powerlawPlot.axis2.plot(
        beats, 
        dist_expon * len(truncated_sorted_size_event) * bmax / nbins,
        '--y', 
        lw=3, 
        label='Exponential')
    analysisGUI.plWindow.powerlawPlot.axis2.plot(
        beats, 
        dist_powerlaw * len(truncated_sorted_size_event) * bmax / nbins,
        '-r',
        lw=3, 
        label='Powerlaw')
    analysisGUI.plWindow.powerlawPlot.axis2.set_xlabel("Event Size (beats)")
    analysisGUI.plWindow.powerlawPlot.axis2.set_ylabel("Number of Events")
    analysisGUI.plWindow.powerlawPlot.axis2.legend()
    analysisGUI.plWindow.powerlawPlot.axis2.set_ylim(0,100)
    analysisGUI.plWindow.powerlawPlot.fig.tight_layout()
    analysisGUI.plWindow.powerlawPlot.draw()

#R/p analysis
def likelihood_and_significance(analysisGUI, pace_maker, batch_data):
    
    # Check whether using translocation data from individual dataset or batch.
    if hasattr(pace_maker, "transloc_events"):
        size_event = pace_maker.transloc_events
        print("Using dataset translocations.")
    elif hasattr(batch_data, "batch_translocs"):
        size_event = batch_data.batch_translocs
        print("Using batch translocations.")
    
    sorted_size_event = sorted(size_event)

    #Fitting Power Law to Data
    PL_results = pl.Fit(sorted_size_event)

    #Comparing Distributions
    R_ln, p_ln = PL_results.distribution_compare(
        'power_law', 'lognormal')
    R_exp, p_exp = PL_results.distribution_compare(
        'power_law', 'exponential', 
        normalized_ratio = True)
    R_weib, p_weib = PL_results.distribution_compare(
        'power_law', 'stretched_exponential')

    #Result Text for R/p Readout
    if p_ln <= 0.05:
        if R_ln >= 0:
            ln_results = "YES, Power law is more likely than a log normal distribution"
        elif R_ln < 0:
            ln_results = "NO, Log normal is more likely than a power law distribution"
    else:
        ln_results = "Cannot reject the null that power law and log normal distributions are equally likely."

    if p_exp <= 0.05:
        if R_exp >= 0:
            exp_results = "YES, Power law is more likely than an exponential distribution"
        elif R_exp < 0:
            exp_results = "NO, Exponential is more likely than a power law distribution"
    else:
        exp_results = "Cannot reject the null that power law and exponential distributions are equally likely."

    if p_weib <= 0.05:
        if R_weib >= 0:
            weib_results = "YES, Power law is more likely than a Weibull distribution"
        elif R_weib < 0:
            weib_results = "NO, Weibull is more likely than a power law distribution"
    else:
        weib_results = "Cannot reject the null that power law and Weibull distributions are equally likely."

    #R/p readout
    complete_rp_readout = [
        "Power Law vs Log Normal Stats:" + "\n",
        " - R value = {}".format(R_ln) + "\n",
        " - p value = {}".format(p_ln) + "\n" + "\n",
        "Results: {}".format(ln_results) + "\n" + "\n" + "\n",
        "Power Law vs Exponential Stats:" + "\n",
        " - R value = {}".format(R_exp) + "\n",
        " - p value = {}".format(p_exp) + "\n" + "\n",
        "Results: {}".format(exp_results) + "\n" + "\n" + "\n",
        "Power Law vs Stretched Exponential Stats:" + "\n",
        " - R value = {}".format(R_weib) + "\n",
        " - p value = {}".format(p_weib) + "\n" + "\n",
        "Results: {}".format(weib_results) + "\n" + "\n" + "\n" + "\n",
        "Parameters:" + "\n",
        "R is the loglikelihood ratio between the two distributions tested." + "\n",
        "A positive R value indicates a power law distribution is a more likely fit for the distribution" + "\n" + "\n",
        "P-value: if below 0.05, we can reject the null hypothesis that both distributions are equally likely."
    ]

    #Display Readout 
    analysisGUI.plWindow.statsPrintout.setPlainText(
        "".join(map(str, complete_rp_readout)))

# End
