# Author: Madelynn Mackenzie
# Contact email: madelynnmack@gmail.com
# Mentor & Co-author: Christopher S. Dunham
# Email: csdunham@chem.ucla.edu, csdunham@protomail.com
# This is an original work unless otherwise noted.

# Function to perform power-law analysis of cardiomyocyte pacemaker 
# translocations.

# Import modules
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
# End import


# Begin distribution comparisons between power law, other distributions.
def compare_distribs(analysisGUI, pace_maker, batch_data, electrode_config, 
beat_amp_int): 
    try:
        # Check whether using translocation data from individual dataset or 
        # batch.
        if hasattr(batch_data, "batch_translocs"):
            transloc_data = batch_data.batch_translocs
            print("Using batch dataset translocations.")
        else:
            print("Using individual recording translocations.")
            # Call translocation detection function for individual dataset
            detect_transloc.pm_translocations(analysisGUI, pace_maker, 
                electrode_config, beat_amp_int)
            transloc_data = pace_maker.transloc_events
            
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
                bin_width = 2 * (
                    spstats.iqr(transloc_data) / (len(transloc_data)**(1/3)))
                # Check if bin_width = 0 to avoid infinity (div by 0) error
                if bin_width == 0:
                    bin_width = 1
                    print("F-D calculated bin_width = 0; using bin_width = 1")
                else:
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
                title=f"bin method: {bin_method}.\n" + 
                    f"n-bins: {nbins}\n" + 
                    f"multi-only: {multi_events}",
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
            compare_via_LLR(analysisGUI, sorted_transloc_data)

        else: 
            print("Insufficient data.")
    except (UnboundLocalError, AttributeError) as no_data_error:
        print("No data.")
    except TypeError:
        print("Cannot plot histogram: no translocations detected.")
# End function.


# Generate probability distribution function (PDF) plots.
def pdf_plotting(analysisGUI, sorted_transloc_data):
    check_discrete = analysisGUI.plWindow.discreteSelect.currentText()
    set_discrete = str2bool(check_discrete)
    ax_pdf = analysisGUI.plWindow.powerlawPlot.axis2
    ax_pdf.cla()

    # Call get_xmin_xmax function to obtain valid values for xmin, xmax
    xmin, xmax = get_xmin_xmax(analysisGUI)
    
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
# End pdf_plotting


# Generate complementary cumulative distribution function (CCDF) plots
def ccdf_plotting(analysisGUI, sorted_transloc_data):
    check_discrete = analysisGUI.plWindow.discreteSelect.currentText()
    set_discrete = str2bool(check_discrete)
    ax_ccdf = analysisGUI.plWindow.powerlawPlot.axis3
    ax_ccdf.cla()

    # Call get_xmin_xmax function to obtain valid values for xmin, xmax
    xmin, xmax = get_xmin_xmax(analysisGUI)
    
    PL_results = pl.Fit(
        sorted_transloc_data, 
        discrete=set_discrete,
        xmin=xmin, 
        xmax=xmax)
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
# End ccdf_plotting


# Compare distributions using log-likelihood ratios.
def compare_via_LLR(analysisGUI, sorted_transloc_data):
    try:  
        check_discrete = analysisGUI.plWindow.discreteSelect.currentText()
        set_discrete = str2bool(check_discrete)

        first_distrib = analysisGUI.plWindow.distribSelect.currentText()
        print(f"First distribution: {first_distrib}")

        # Call get_xmin_xmax function to obtain valid values for xmin, xmax
        xmin, xmax = get_xmin_xmax(analysisGUI)
        
        # Fitting Power Law to Data
        PL_results = pl.Fit(
            sorted_transloc_data, 
            discrete=set_discrete,
            xmin=xmin, 
            xmax=xmax)

        # Parameters calculated from powerlaw module:
        # Power law
        PL_alpha = PL_results.power_law.alpha
        PL_sigma = PL_results.power_law.sigma
        PL_xmin = PL_results.power_law.xmin
        # Exponential
        exp_lamda = PL_results.exponential.Lambda
        # Lognormal
        lognorm_mu = PL_results.lognormal.mu
        lognorm_sigma = PL_results.lognormal.sigma
        # Weibull
        weibull_lamda = PL_results.stretched_exponential.Lambda
        weibull_beta = PL_results.stretched_exponential.beta
        # Truncated power law
        trunc_alpha = PL_results.truncated_power_law.alpha
        trunc_lamda = PL_results.truncated_power_law.Lambda
        trunc_xmin = PL_results.truncated_power_law.xmin

        #use to see what names to use for comparisons to other distributions
        distributions = list(
            PL_results.supported_distributions.keys())
        distributions = [
            distrib for distrib in distributions if distrib != first_distrib]
        
        summary_of_LLRs = [
            "---------------Parameter Summary---------------\n",
            f"xmin = {PL_xmin}\n",
            f"xmax = {xmax}\n",
            "Power law \u03b1 = " + f"{PL_alpha}\n",
            "Power law error (\u03c3) = " + f"{PL_sigma}\n",
            "Exponential \u03bb = " + f"{exp_lamda}\n",
            "Log-normal \u03bc = " + f"{lognorm_mu}\n",
            "Log-normal \u03c3 = " + f"{lognorm_sigma}\n",
            "Weibull \u03bb = " + f"{weibull_lamda}\n",
            "Weibull \u03b2 = " + f"{weibull_beta}\n",
            "Trunc. power law \u03b1 = " + f"{trunc_alpha}\n",
            "Trunc. power law \u03bb = " + f"{trunc_lamda}\n\n",
            "-----Log-likelihood Ratio (LLR) Evaluation-----\n"]

        for distrib in distributions:
            # R = loglikelihood ratio between the two candidate distributions
            # positive if the data is more likely in the first distribution 
            # negative if the data is more likely in the second distribution
            # p = significance value for that direction
            # normalized ratio option: normalizes R by standard deviation 
            # (used to calc p)
            LLR, pval = PL_results.distribution_compare(
                first_distrib, 
                distrib, 
                normalized_ratio=True)

            temp_LLR = f"LLR of {first_distrib}:{distrib} = {LLR}\n"
            temp_pval = f"p-value of {first_distrib}:{distrib} = {pval}\n\n"
            summary_of_LLRs.append(temp_LLR)
            summary_of_LLRs.append(temp_pval)

            print(f"Ratio test of {first_distrib}:{distrib} = {LLR}")
            print(f"P-value of {first_distrib}:{distrib} = {pval}\n\n")
        
        # Display Readout 
        analysisGUI.plWindow.statsPrintout.setPlainText(
            "".join(map(str, summary_of_LLRs)))

    except UnboundLocalError:
        print("Unbound error occurred.")
    except TypeError:
        print("Cannot calculate statistics: no translocations detected")
# End compare_via_LLR


# Function to compare true/false strings from drop-down boxes for True condition
def str2bool(check_string):
    return str(check_string).lower() in ("true")
# End str2bool function


# Function to run checks for values of xmin, xmax
def get_xmin_xmax(analysisGUI):
    try:
        # Get xmin user input
        xmin = analysisGUI.plWindow.xminEdit.text()
        if xmin == "":
            xmin = None
            print("No xmin given. Letting powerlaw determine xmin.")
        elif float(xmin) < 1.0:
            print("xmin must be greater than or equal to 1. Defaulting to 1")
            xmin = 1

        # Get xmax user input
        xmax = analysisGUI.plWindow.xmaxEdit.text()
        if xmax == "":
            xmax = None
            print("No xmax chosen. Defaulting to 'None'.")
        elif xmin != None and float(xmax) < float(xmin):
            print(f"xmax must be larger than xmin. Defaulting to {xmin + 150}")
            xmax = xmin + 100
        elif float(xmax) < 1.0:
            print("xmax must be positive and greater than or equal to 1." + 
                "Taking absolute value.")
            xmax = abs(xmax)
        # Return xmin, xmax values after input checks
        return xmin, xmax
    except ValueError: 
        print("Letter (character) inputs are not permitted for xmin, xmax.")
# End get_xmin_xmax function.

# End
