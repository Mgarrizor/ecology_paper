Instructions on the use and analysis of the simulations run from the source code (main_competition.py, ios.py, params_competition.py, tstepping_competition.py)

Main =======================================
main_competition.py  

Modules ====================================

tstepping_competition.py: time stepping algorithm

ios.py:                   creates the directories where the output data will be saved. These are 1) the simulation parameters; 2) The population densities of the different species as 
                          a function of time and space and the fraction of the spatial system occupied by each.

params_competition.py:    module that controls the simulation parameters, e.g., integration time or the number of habitats

Execute sumulation: python3 main_competition.py 



