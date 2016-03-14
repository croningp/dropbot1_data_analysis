In this folder are all the tools needed to extract information from the data folder into coherent, usable, datasets.

The ```generate_datasets.py``` will, for each folder in the ```data``` folder, collect all experiments, create a new folder of the same name and save 5 files into it:

- ```info.json``` is a json formatted file containing basic info such as the number of experiments and the name of the parameters or features used.

- ```path.json``` is a json formatted file containing the path of each experiments ordered as in the csv files

- ```x.csv``` is a csv file, formatted such that numpy can read it easily, containing all the parameters in a matrix form. Each line represent one experiment.

- ```y.csv``` is a csv file, formatted such that numpy can read it easily, containing all the features in a matrix form. Each line represent one experiment.

- ```full.csv``` is a csv file containing all the information from ```path.json```, ```x.csv```, and ```y.csv```, including a column header.

All lines in the csv files correspond to the same experiment.
