# Evry Schatzman school on stellar physics -- 05/10/2021
## An introduction on transmission spectroscopy with SPIRou


###  Getting Started

To run this tutorial on your local machine, please download this code and associated model/data on your local machine. Using **jupyter notebook**, open the code **main.ipynb** and follow the instructions therein. The code relies on 3 ".py" files: **src.py** (main functions for the data reduction), **plots.py** (functions to plot data at different steps on the process and **correlation.py** to compute a correlation between the data and a planet atmosphere template for different planet orbital parameters.



### Prerequisites

The following programs are needed to make the code work:
1 - python3 (code implemented on python3 - any version > 3.5)
2 - jupyter notebook 
    Installed via pip: pip install jupyterlab
    or conda. See https://jupyter.org/install for proper installation procedure
    To run the notebook open a terminal in the github directory containing this tutorial (downloaded)
    Type 'jupyter notebook' command to run the module. It opens a new window in your browser.
    Open "Main.ipynb" and start the tutorial.
    
The following python modules are required (in addition to standard python modules)
  - scipy
  - astropy
  - pandas

### Authors

* **Baptiste Klein** - Department of Physics, University of Oxford 
* **Benjalin Charnay** - LESIA, Observatoire de Paris, Meudon, France
* **Florian Debras** - IRAP, Toulouse, France
