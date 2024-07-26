**This is an in-development tool for test purposes only. This tool is not currently tested for clinical use.**

Please see requirements.txt for installation of required dependencies.

This repository provides scripts for the development of a semantic textual similarity model to link common symptom inventories used in TBI research. It also contains codes for development of a web tool. 

The code for the development of linked STS scores across inventories is available in the script `STS_score.py`. The crosswalk model using these STS scores is available in `crosswalk_symptom_inventories.py`.
`app.py` implements a local browser version of the online web tool using the command `shiny run app.py`. After any changes to the web tool, please run `shiny run --reload --launch-browser your_dir/app.py` and reload the browser, where "your_dir" is the folder of the application.
Full documentation of the tool properties, accuracy, and development are available online in the associated publication: TBD

A shiny app hosting this application is available [here](https://enigma-tools.shinyapps.io/symptom-inventories-calculator/)

Instructions for the web tool:
1) Select the inventory input and output from the dropdown menu. Then input the numbers into the fields (the ranges are specified below the inventory name).
2) Run conversion: Click the "Convert table" button to generate the converted scores.
3) Download converted data: There is also a 'Download Conversion' button to download the converted table as a csv.
