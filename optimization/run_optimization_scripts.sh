#!/bin/bash

# this fisrt and only argument is the definition of the stl export, it defines the $fn variable in openscad
if [ -z "$1" ]
  then
    nrepeats=100
    echo "No argument supplied -> nrepeats=$nrepeats"
  else
    nrepeats=$1
fi

python estimate_best_ea_params.py $nrepeats
python plot_ea_methods_comparison.py $nrepeats
python plot_GA_comparison.py $nrepeats
python plot_cmaes_comparison.py $nrepeats
