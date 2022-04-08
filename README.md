# cardio_pymea
Repository for Cardio PyMEA, a Python application for the analysis of microelectrode array (MEA)-derived cardiomyocyte data.

## Author Information <br>
Name: Christopher S. Dunham <br>
Institution: University of California, Los Angeles<br>
Email: csdunham@chem.ucla.edu<br>
<br>

## General Information <br>
Cardio PyMEA is a Python application designed for the analysis of microelectrode array (MEA)-derived cardiomyocyte data. Its key functionalities are highlighted in the preprint posted on biorXiv (https://www.biorxiv.org/content/10.1101/2022.03.25.485780v1) and will hopefully achieve publication in the near future. This README will be updated again at that time.  It currently works for data obtained from Multichannel Systems MEA hardware (e.g. MEA2100) using the MC_Rack software (data acquisition) and MC_Data tool (conversion of <i>*.mcd</i> files to <i>*.txt</i>. However, it can be readily expanded for use with other MEA geometries and configurations, provided that the input file follows a certain format and MEA electrode geometries are provided. This information is described further in the S1 File ("Tutorial") from the publication associated with this software (DOI to be provided at a later date, if accepted for publication).
<br>
<br>

## Operation of Cardio PyMEA <br>
Below is a brief description of how to operate this application.

First, check to make sure you are running Python 3.10 or higher. If not, please install a suitable version of Python to your system.
Then, you should generate a Python virtual environment and use the requirements.txt file, in combination with pip, to load all of the necessary Python dependencies.

Next, source your virtual environment and run cardio_pymea.py in the terminal. Once Cardio PyMEA is running, you may individually analyze data files 
(<i>*.txt</i> format as outputted from the MC_Rack and MC_Data programs written by Multichannel Systems) by choosing File --> Import.

Once the import is completed, enter your chosen parameters for minimum peak height and minimum peak distance. 
Additionally, you may apply data truncation if you wish to view a narrow window of time in your data set, or manually exclude electrodes from analysis, using the checkboxes for each function and inputting the recording range or selecting the electrode(s) to exclude, respectively.

After setting your input parameters, select Calculations --> Find Beats. This operation will detect all of the beats in the recording.
Next, you may perform your chosen calculations from the Calculations menu. 
"All" encompasses all calculations except for Field Potential Duration. 
Field potential duration relies on the outputs of some of properties calculated using "All". 
If you want that output, you will need to click on it after running "All". 
Next, if you want to evaluate the data for pacemaker translocations, select Tools --> Detect Translocations.
If there are translocations in the data set, you will see the quiescent period lengths printed in the terminal.
If there are none, Cardio PyMEA will tell you that there are none.


For batch processing, format the batch spreadsheet file (File 19) for your data and data locations.
Then, after launching Cardio PyMEA, select File --> Batch. Choose your batch spreadsheet file. 
Cardio PyMEA will iterate through all of the files in the batch spreadsheet file and identify all pacemaker translocations in the provided data.


Once you have translocations to analyze, select Statistics -> Power Law Distribution Comparison. 
This will perform the statistical analysis and show the ratio test and p-value outputs for comparisons between power law and alternative heavy-tailed distributions. 
For more information and context regarding this subject, please see the related work at the following DOI: 10.1371/journal.pone.0263976.

To make changes to Cardio PyMEA's MEA configurations, please see S1 File ("Tutorial") from the publication associated with this software (DOI to be provided at a later date, if accepted for publication).
