# MaxDivSec

This repo contains the code used to produce the results found for the MaxDivSec instance selection method. The project is divided like this:

* data: The folder with the raw data, interim data (cross-validation sets) and final results.
* docs: The folder for the github page
* figures: The folder with all the figures
* src: The folder with the code for the experiment which contains the following folder
    * cross_validation: The code for the creation of the cross-validation sets
    * data: The code to read, process and analyse the data
    * instance_selection_methods: The code for the mahalanobis outlier detection method
    * models: The code for the KNN training and prediction
    * not in a folder: Those are the script to get the various results. Their working directory must be MaxDivSec and
     not MaxDivSec/src. Each script have their parameters in comment at the top.

For the MaxDivSec method, we first created the distance matrix with OBMA_distance_file.py. That also creates the shell
scrip to run the OBMA solver with the correct parameters. Once the OBMA solver is done, we use the OBMA_train.py file to get the results.
