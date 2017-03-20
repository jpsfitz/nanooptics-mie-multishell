# nanooptics-mie-multishell

# Getting jupyter up and running on Ubuntu
These installation instructions are for Linux, but they will roughly work for Windows as well with a few adjustments.


## A. Download and install anaconda. Do not give root priveleges.
Anaconda provides the easiest way to get a jupyter notebook environment up and running. Follow the directions at  
https://www.continuum.io/downloads
Follow instructions there. 
* conda and jupyter work best as a local installation without root priveleges. 
* If you have ipython and jupyter installed through Ubuntu, there will be problems. Uninstall it first.
* Afterwards, try not to install packages with 'pip' but instead with 'conda'.
* The command line reference: https://conda.io/docs/commands.html 

## B. Install conda-forge channel
The 'conda-forge' channel has a lot of very useful and often more up-to-date packages.  
https://conda-forge.github.io/    
`conda config --add channels conda-forge`  
* After installation, browse the newly available packages at  https://conda-forge.github.io/feedstocks
* You probably want to update conda now. There are often conflicts.   
`conda update --all`  

## C. Notebook extensions from conda-forge & github
These notebook extensions make jupyter notebooks much easier to read and use.  
https://github.com/conda-forge/jupyter_contrib_nbextensions-feedstock  
https://github.com/ipython-contrib/jupyter_contrib_nbextensions  
`conda install jupyter_contrib_nbextensions`   
`jupyter contrib nbextension install --user`
* Now you can use a tab in the jupyter environment (jupyter_nbextensions_configurator) to enable these extensions.
* There are several handy extensions, but the favorites are 
  - Table of Contents (2)
  - Hide input all
  - Collapsible Headings
  - Execute Time
  - Hide input

## D. Parallel computing python support
This will allow your python jupyter notebooks to access a pool of python kernels, which can be controlled from within the notebook.
https://github.com/ipython/ipyparallel  
https://github.com/conda-forge/ipyparallel-feedstock  
https://ipyparallel.readthedocs.io/en/latest/  
`conda install ipyparallel`    
`ipcluster nbextension enable --user`  
`jupyter serverextension enable --py ipyparallel`  
`jupyter nbextension install --py ipyparallel --user`  
`jupyter nbextension enable ipyparallel --user --py`  

## E. Some customization
http://jupyter-notebook.readthedocs.io/en/latest/config.html  
Create a config file  
`jupyter notebook --generate-config`  
Edit the generated config file (`~/.jupyter/jupyter_notebook_config.py`)
- Change to use non-default browser  
  - `c.NotebookApp.browser = '/usr/binfirefox'`
  - make sure that it is not indented.
- Change to set default opening directory 
  - `c.NotebookApp.notebook_dir = '/home/fit/Dropbox/python`
- May need to delete `.json` files in `/home/fit/.jupyter` (fixes conda tab)

## F. Install Peak Utils
PeakUtils automates peak fitting in noisy or sloped curves.  
http://pythonhosted.org/PeakUtils/  
https://pypi.python.org/pypi/PeakUtils  
Follow directions in readme.

## G. Some more customization
### Inline plots
Default setting for plots is to pop out of notebook. Change the iPython config files to have the plots in-line.  
`ipython profile create`  
In `~/.ipython/profile_default/ipython_kernel_config.py` change:  
`c.InteractiveShellApp.matplotlib = "inline"`  
* again, make sure there is no indentation.  

### Size and fonts of inline plots
Add to `~/.ipython/profile_default/ipython_kernel_config.py`:   
```
# Subset of matplotlib rcParams that should be different for the inline backend.
c.InlineBackend.rc = {'font.size': 12, 'figure.figsize': (5.0, 5.0), 'figure.facecolor': 'white', 'savefig.dpi': 144, 'figure.subplot.bottom': 0.125, 'figure.edgecolor': 'white'}
```
