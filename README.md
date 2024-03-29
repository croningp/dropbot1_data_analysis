

##

This set of tools has been tested under Python 2.7.6 on Ubuntu 14.04 LTS. Despite all our efforts, we cannot guarantee everything will be executable on other OS or Python version.

## Dependencies

Aside from the standard libraries, we are using the following libraries. You do not have to install them all, it depends on the task you are performing.

- [opencv](http://opencv.org/): Image analysis with python binding.
Version: cv2.__version__ is '2.4.8'

- [numpy](http://www.numpy.org/): Scientific computing in Python.
Version: numpy.__version__ is '1.10.4'

- [scipy](http://www.scipy.org/scipylib/index.html): More scientific computing in Python.
Version: scipy.__version__ is '0.16.1'

- [sklearn](http://scikit-learn.org/): Machine Learning in Python.
Version: sklearn.__version__ is '0.16.1'

- [gmr](https://github.com/AlexanderFabisch/gmr): A library to do conditional inference with Gaussian Mixture Models.
Version: gmr.__version__ is '1.1'

- [joblib](https://pythonhosted.org/joblib/): Easy parallel computing.
See also: https://github.com/joblib/joblib
Version: joblib.__version__ is '0.9.1'

- [ternary](https://github.com/marcharper/python-ternary): Ternary plotting for python with matplotlib.
Commit: 70ac90d44ffc88837fb7b5df701143b23c58b3a4

- [seaborn](https://github.com/mwaskom/seaborn): Statistical data visualization using matplotlib.
Version: seaborn.__version__ is '0.7.0'

- [jupyter notebook](http://jupyter.org/): Interactive data science and scientific computing.
Version: jupyter --version is 4.0.6

## Download the raw data

Link to files

## Video analysis and feature extraction

Run the script [analyse_all_videos.py](feature_extraction/analyse_all_videos.py) in the [feature_extraction](feature_extraction) folder. This script will store a ```features.json``` along each ```video.avi```. See [feature_extraction](feature_extraction) folder for more info.

## Generating datasets

Run the script [generate_datasets.py](datasets/generate_datasets.py) in the [datasets](datasets) folder. This script will collect all experiments from the [data](data) folder and compile them into easy to use csv files. See [datasets](datasets) folder for more info.


## Regression

Regression algorithms allow to fit a model on the datasets. We used state of the art, well established, regression algorithms implemented in the [sklearn](http://scikit-learn.org/) library.

Run the script [cv_train_regressionn.py](models/regression/cv_train_regressors.py) in the [models/regression](models/regression/) folder. This script will select the best set of parameters for many different regression algorithm and store the best ones in the [pickled](models/regression/pickled/) folder. See [models/regression](models/regression/) folder for more info.

The script [plot_ternary.py](models/regression/plot_ternary.py) plots, for each regressors model, the heat maps of fitnesses given oils composition. They are stored in the [plot](models/regression/plot) folder.

The script [print_regressors_info.py](models/regression/print_regressors_info.py) prints all the information about the regressors training and performances.

## Mixture

Regression model we used cannot model the stochasticity of the data. A Gaussian Mixture Model (GMM) is more suited for that purpose.

We first estimate the 'optimal' number of Gaussians to use. The script [n_component_estimation.py](models/mixture/n_component_estimation.py) estimate the [BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion) for models ranging from 1 to 100 Gaussians. We estimate this for each dimension separately, as well as for all the dimension together. The result are stored  in  the  [csv](models/mixture/csv) folder. The [plot_bics.py](models/mixture/plot_bics.py) script subsequently plots the bic scores and stores them in the [plot/bics](models/mixture/plot/bics/) folder

Then, given the bic scores, we select the number of Gaussians giving the lowest bic score and train the corresponding GMM. The script [train_gmms.py](models/mixture/train_gmms.py) does just that and store the resulting model in the [pickled](models/mixture/pickled/) folder.
