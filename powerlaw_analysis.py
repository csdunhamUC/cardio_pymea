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
import scipy.stats as spstats
import statsmodels as sm
import powerlaw as pl
import detect_transloc
import re


# data: list, nbins: int = 50, multi_events: bool = False, bin_method: str = "none"
def compare_distribs(analysisGUI, pace_maker, batch_data): 
    try:
        # Check whether using translocation data from individual dataset or 
        # batch.
        if hasattr(batch_data, "batch_translocs"):
            transloc_data = batch_data.batch_translocs
            print("Using dataset translocations.")
        elif hasattr(pace_maker, "transloc_events"):
            transloc_data = pace_maker.transloc_events
            # size_event = batch_data.batch_translocs
            print("Using batch translocations.")

        check_multi = analysisGUI.plWindow.multiSelect.currentText()
        multi_events = str2bool(check_multi)
        bin_method = analysisGUI.plWindow.binMethodSelect.currentText()

        # Check for sufficient data.
        # Only conduct analysis if there's at least 5 translocations
        if len(transloc_data) > 5: 
            # If toggled, filter for multiple events only.
            if multi_events == True:
                # Find unique values, counts in translocation data
                unique_vals, unique_counts = np.unique(transloc_data, 
                    return_counts=True)
                # Find repeating values only (i.e. count greater than 1)
                repeat_vals_only = unique_vals[np.where(unique_counts > 1)]
                repeat_counts_only = unique_counts[np.where(unique_counts > 1)]
                # Filter data for those events that occur multiple times.
                temp_data = [
                    val for val in transloc_data if val in repeat_vals_only]
                # Re-assign transloc_data to the filtered data
                transloc_data = temp_data
                # Delete the temporary variable
                del temp_data
            # Check method from drop-down menu for bin method selection
            if bin_method == "Manual Entry" or bin_method == "none":
                nbins = int(analysisGUI.nbinsEdit.text())
            # Use Sturge's Rule to determine nbins
            elif bin_method == "Sturge's Rule":
                nbins = int(np.ceil(1 + 3.322*np.log(len(transloc_data))))
            # Use Freedman-Diaconis Rule to determine nbins.
            elif bin_method == "Freedman-Diaconis":
                bin_width - 2 * (
                    spstats.iqr(transloc_data) / (len(transloc_data)**(1/3)))
                nbins = int(
                    np.ceil((max(transloc_data) - 
                        min(transloc_data)) / bin_width))

            sorted_transloc_data = np.sort(transloc_data)
            
            ###################################################################
            # Model Distributions for Comparison
            # Exponential distribution
            exp_dist_batch = sp.stats.expon
            args_exp_batch = exp_dist_batch.fit(sorted_transloc_data)

            # Power-law distribution
            pl_dist_batch = sp.stats.powerlaw
            args_pl_batch = pl_dist_batch.fit(sorted_transloc_data)

            # Weibull distribution
            weib_dist_batch = sp.stats.weibull_min
            args_weib_batch = weib_dist_batch.fit(sorted_transloc_data)

            # Log-normal distribution
            lognorm_dist_batch = sp.stats.lognorm
            args_lognorm_batch = lognorm_dist_batch.fit(sorted_transloc_data)

            # Placeholder for x-axis for PDF plots.
            temp_events = np.linspace(
                0, 
                sorted_transloc_data.max(), 
                len(sorted_transloc_data))

            exp_dist_vals = exp_dist_batch.pdf(
                temp_events, *args_exp_batch)
            pl_dist_vals = pl_dist_batch.pdf(
                temp_events, *args_pl_batch)
            weib_dist_vals = weib_dist_batch.pdf(
                temp_events, *args_weib_batch)
            lognorm_dist_vals = lognorm_dist_batch.pdf(
                temp_events, *args_lognorm_batch)

            ###################################################################
            # Plotting.

            # Clear axis if previously plotted.
            analysisGUI.plWindow.powerlawPlot.axis1.cla()
            
            # Plot histogram.
            analysisGUI.plWindow.powerlawPlot.axis1.hist(
                sorted_transloc_data, 
                nbins, 
                color="gray", 
                ec="black", 
                lw=0.5, 
                alpha=0.2)
            
            analysisGUI.plWindow.powerlawPlot.axis1.set(
                title=f"Histogram Assessment of PDF Fits\n" + 
                f"Bin method: {bin_method}. n-bins: {nbins}\n" + 
                    f"Multi-only: {multi_events}",
                xlabel="Quiescent Period (beats)",
                ylabel="Number of Events",
                ylim=(0, 60)) # need to refine this to dynamically change

            # Plot distributions.
            analysisGUI.plWindow.powerlawPlot.axis1.plot(
                temp_events, 
                (lognorm_dist_vals * len(sorted_transloc_data) 
                    * sorted_transloc_data.max()) / nbins,
                '--',
                color="yellow",
                lw=1, 
                label='Log-Normal')
            analysisGUI.plWindow.powerlawPlot.axis1.plot(
                temp_events, 
                (weib_dist_vals * len(sorted_transloc_data) 
                    * sorted_transloc_data.max()) / nbins,
                '--',
                color="green",
                lw=1, 
                label='Weibull')
            analysisGUI.plWindow.powerlawPlot.axis1.plot(
                temp_events, 
                (exp_dist_vals * len(sorted_transloc_data) 
                    * sorted_transloc_data.max()) / nbins,
                '--', 
                color="red",
                lw=1, 
                label='Exponential')
            analysisGUI.plWindow.powerlawPlot.axis1.plot(
                temp_events, 
                (pl_dist_vals * len(sorted_transloc_data) 
                    * sorted_transloc_data.max()) / nbins,
                '--',
                color="blue",
                lw=1, 
                label='Power Law')

            # Generate legend, clean up plot, display.
            analysisGUI.plWindow.powerlawPlot.axis1.legend()
            analysisGUI.plWindow.powerlawPlot.fig.tight_layout()
            analysisGUI.plWindow.powerlawPlot.draw()

            # Call remaining functions to plot PDF, CCDF, and 
            # generate LLR report.
            pdf_plotting(analysisGUI, sorted_transloc_data)
            ccdf_plotting(analysisGUI, sorted_transloc_data)
            # compare_via_LLR(analysisGUI, sorted_transloc_data)

        else: 
            print("Insufficient data.")
    except UnboundLocalError:
        print("No data.")
    except TypeError:
        print("Cannot plot histogram: no translocations detected.")


def pdf_plotting(analysisGUI, sorted_transloc_data):
    check_discrete = analysisGUI.plWindow.discreteSelect.currentText()
    set_discrete = str2bool(check_discrete)
    ax_pdf = analysisGUI.plWindow.powerlawPlot.axis2
    ax_pdf.cla()

    xmin = analysisGUI.plWindow.xminEdit.text()
    if xmin == "":
        xmin = None
        print("No xmin given. Letting powerlaw determine xmin.")
    elif float(xmin) < 1.0:
        print("xmin must be greater than or equal to 1. Defaulting to 1")
        xmin = 1

    xmax = analysisGUI.plWindow.xmaxEdit.text()
    if xmax == "":
        xmax = None
        print("No xmax chosen.")
    elif xmin != None and float(xmax) < float(xmin):
        print("xmax must be larger than xmin. Defaulting to xmin + 150")
        xmax = xmin + 100
    elif float(xmax) < 1.0:
        print("xmax must be positive and greater than or equal to 1." + 
            "Taking absolute value.")
        xmax = abs(xmax)
    
    PL_results = pl.Fit(
        sorted_transloc_data, 
        discrete=set_discrete,
        xmin=xmin, 
        xmax=xmax)
    pdf_alpha = PL_results.alpha

    PL_results.plot_pdf(
        color = 'black',
        linewidth = 2,
        label=f"Empirical Data\nx$_{{min}}$ = {PL_results.xmin}",
        ax=ax_pdf,
        alpha=0.8)
    PL_results.power_law.plot_pdf(
        color = 'blue',
        linestyle = '--',
        label=f"Power Law\n" + r"$\alpha$ = " + f"{-1*pdf_alpha:.2f}",
        ax=ax_pdf)
    PL_results.lognormal.plot_pdf(
        color = 'yellow',
        linestyle = '--',
        label=f"Log-Normal",
        ax=ax_pdf)
    PL_results.exponential.plot_pdf(
        color = 'r',
        linestyle = '--',
        label=f"Exponential",
        ax=ax_pdf)
    PL_results.stretched_exponential.plot_pdf(
        color = 'green',
        linestyle = '--',
        label=f"Weibull",
        ax=ax_pdf)

    ax_pdf.set(
        xlabel="Quiescent Period (beats)",
        ylabel="p(X)",
        xlim=(10**0, 10**2),
        ylim=(10**-4, 10**0.5))
 
    analysisGUI.plWindow.powerlawPlot.fig.tight_layout()
    analysisGUI.plWindow.powerlawPlot.draw()


def ccdf_plotting(analysisGUI, sorted_transloc_data):
    check_discrete = analysisGUI.plWindow.discreteSelect.currentText()
    set_discrete = str2bool(check_discrete)
    ax_ccdf = analysisGUI.plWindow.powerlawPlot.axis3
    ax_ccdf.cla()


    PL_results = pl.Fit(sorted_transloc_data, discrete=set_discrete)
    ccdf_alpha = PL_results.alpha

    # CCDF
    PL_results.plot_ccdf(
        color = 'black',
        linewidth = 2,
        label=f"Empirical Data\nx$_{{min}}$ = {PL_results.xmin}",
        ax=ax_ccdf,
        alpha=0.8)
    PL_results.power_law.plot_ccdf(
        color = 'blue',
        linestyle = '--',
        label=f"Power Law\n" + r"$\alpha$ = " + f"{-1*ccdf_alpha:.2f}",
        ax=ax_ccdf)
    PL_results.lognormal.plot_ccdf(
        color = 'yellow',
        linestyle = '--',
        label=f"Log-Normal",
        ax=ax_ccdf)
    PL_results.exponential.plot_ccdf(
        color = 'r',
        linestyle = '--',
        label=f"Exponential",
        ax=ax_ccdf)
    PL_results.stretched_exponential.plot_ccdf(
        color = 'green',
        linestyle = '--',
        label=f"Weibull",
        ax=ax_ccdf)
    
    ax_ccdf.set(
        xlabel="Quiescent Period (beats)",
        ylabel=r"p(X$\geq$x)",
        xlim=(10**0, 10**2),
        ylim=(10**-4, 10**0.5))

    analysisGUI.plWindow.powerlawPlot.fig.tight_layout()
    analysisGUI.plWindow.powerlawPlot.draw()


# Compare distributions using log-likelihood ratios.
def compare_via_LLR(analysisGUI, sorted_transloc_data):
    try:  
        if len(size_event) > 5:
            sorted_size_event = sorted(size_event)
            
            check_discrete = analysisGUI.plWindow.discreteSelect.currentText()
            set_discrete = str2bool(check_discrete)

            first_distrib = analysisGUI.plWindow.distribSelect.currentText()
            print(f"First distribution: {first_distrib}")

            # Fitting Power Law to Data
            PL_results = pl.Fit(
                sorted_transloc_data, 
                discrete=True)

            # Comparing Distributions
            R_ln, p_ln = PL_results.distribution_compare(
                'power_law', 'lognormal',
                normalized_ratio=True)
            R_exp, p_exp = PL_results.distribution_compare(
                'power_law', 'exponential', 
                normalized_ratio=True)
            R_weib, p_weib = PL_results.distribution_compare(
                'power_law', 'stretched_exponential',
                normalized_ratio=True)
            R_dtpl, p_dtpl = PL_results.distribution_compare(
                "power_law", "truncated_power_law",
                normalized_ratio=True)

            # Result Text for R/p Readout
            if p_ln <= 0.05:
                if R_ln >= 0:
                    ln_results = "YES, Power law is more likely than a log normal distribution"
                elif R_ln < 0:
                    ln_results = "NO, Log normal is more likely than a power law distribution"
            else:
                ln_results = "Sign of R is not statistically significant."

            if p_exp <= 0.05:
                if R_exp >= 0:
                    exp_results = "YES, Power law is more likely than an exponential distribution"
                elif R_exp < 0:
                    exp_results = "NO, Exponential is more likely than a power law distribution"
            else:
                exp_results = "Sign of R is not statistically significant."

            if p_weib <= 0.05:
                if R_weib >= 0:
                    weib_results = "YES, Power law is more likely than a Weibull distribution"
                elif R_weib < 0:
                    weib_results = "NO, Weibull is more likely than a power law distribution"
            else:
                weib_results = "Sign of R is not statistically significant."

            # R/p readout
            complete_rp_readout = [
                "Power Law vs Log Normal:" + "\n",
                " - R value = {}".format(R_ln) + "\n",
                " - p value = {}".format(p_ln) + "\n" + "\n",
                "Results: {}".format(ln_results) + "\n" + "\n" + "\n",
                "Power Law vs Exponential:" + "\n",
                " - R value = {}".format(R_exp) + "\n",
                " - p value = {}".format(p_exp) + "\n" + "\n",
                "Results: {}".format(exp_results) + "\n" + "\n" + "\n",
                "Power Law vs Stretched Exponential:" + "\n",
                " - R value = {}".format(R_weib) + "\n",
                " - p value = {}".format(p_weib) + "\n" + "\n",
                "Results: {}".format(weib_results) + "\n" + "\n" + "\n" + "\n",
                "Parameters:" + "\n",
                "R is the log likelihood ratio between the two distributions tested." + "\n",
                "A positive R value indicates a power law distribution is a more likely fit for the distribution" + "\n" + "\n",
                "P-value: if below 0.05, we can conclude the sign of R is significant."
            ]

            # Display Readout 
            analysisGUI.plWindow.statsPrintout.setPlainText(
                "".join(map(str, complete_rp_readout)))
        else:
            print("Insufficient Data")

    except UnboundLocalError:
        print()
    except TypeError:
        print("Cannot calculate statistics: no translocations detected")


# Function to compare true/false strings from drop-down boxes for True condition
def str2bool(check_string):
    return str(check_string).lower() in ("true")


#def pl_truncated_histogram_plotting(analysisGUI, pace_maker, batch_data): 
#    try:
#        # Check whether using translocation data from individual data or batch.
#        if hasattr(pace_maker, "transloc_events"):
#            size_event = pace_maker.transloc_events
#            print("Using dataset translocations.")
#        elif hasattr(batch_data, "batch_translocs"):
#            size_event = batch_data.batch_translocs
#            # size_event = batch_data.batch_translocs
#            print("Using batch translocations.")

#        if len(size_event) > 5:
            
#            sorted_size_event = sorted(size_event)
            
#            #Removing data below x_min
#            PL_results = pl.Fit(sorted_size_event, discrete=True)
#            truncated_sorted_size_event = [
#                x for x in sorted_size_event if x >= PL_results.power_law.xmin]
            
#            #Model Distributions for Comparison
#            pldist = sp.stats.powerlaw
#            args_powerlaw = pldist.fit(truncated_sorted_size_event)

#            expondist = sp.stats.expon
#            args_expon = expondist.fit(truncated_sorted_size_event)

#            weibulldist = sp.stats.exponweib
#            args_exponweib = weibulldist.fit(truncated_sorted_size_event)

#            lognormdist = sp.stats.lognorm
#            args_lognorm = lognormdist.fit(truncated_sorted_size_event)

#            #Finding the max x value and linerizing the model data
#            bmax = max(truncated_sorted_size_event)
#            beats = np.linspace(0., bmax, 100)

#            #Fitting Distributions to Data
#            dist_powerlaw = pldist.pdf(beats, *args_powerlaw)
#            dist_lognormal = lognormdist.pdf(beats, *args_lognorm)
#            dist_exponweib = weibulldist.pdf(beats, *args_exponweib)
#            dist_expon = expondist.pdf(beats, *args_expon)

#            #Plotting Distributions
#            analysisGUI.plWindow.powerlawPlot.axis2.cla()

#            x_min = PL_results.power_law.xmin
#            print(f"x_min: {x_min}")

#            nbins = 50
#            length = len(sorted_size_event)

#            analysisGUI.plWindow.powerlawPlot.axis2.hist(
#                truncated_sorted_size_event, 
#                nbins)

#            analysisGUI.plWindow.powerlawPlot.axis2.plot(
#                beats, 
#                dist_lognormal * len(truncated_sorted_size_event) * bmax / nbins,
#                '--y', 
#                lw=1, 
#                label='Log Normal')
#            analysisGUI.plWindow.powerlawPlot.axis2.plot(
#                beats, 
#                dist_exponweib * len(truncated_sorted_size_event) * bmax / nbins,
#                '--g', 
#                lw=1, 
#                label='Weibull')
#            analysisGUI.plWindow.powerlawPlot.axis2.plot(
#                beats, 
#                dist_expon * len(truncated_sorted_size_event) * bmax / nbins,
#                '--r', 
#                lw=1, 
#                label='Exponential')
#            analysisGUI.plWindow.powerlawPlot.axis2.plot(
#                beats, 
#                dist_powerlaw * len(truncated_sorted_size_event) * bmax / nbins,
#                '-b',
#                lw=1, 
#                label='Powerlaw')
#            analysisGUI.plWindow.powerlawPlot.axis2.set(
#                title = "X values below {0:.2f} removed".format(x_min),
#                xlabel="Event Size (beats)", 
#                ylabel="Number of Events" )
#            analysisGUI.plWindow.powerlawPlot.axis2.legend()
#            analysisGUI.plWindow.powerlawPlot.axis2.set_ylim(0,60)
#            analysisGUI.plWindow.powerlawPlot.fig.tight_layout()
#            analysisGUI.plWindow.powerlawPlot.draw()
#        else: 
#            print("Insufficient Data")
#    except UnboundLocalError:
#        print()
#    except TypeError:
#        print("Cannot Plot Truncated Histogram: No Translocations Detected")
