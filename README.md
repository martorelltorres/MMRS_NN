This repo works under Ubuntu 20.04 using Tensorflow and Keras.
Installation procedure:


1) sudo apt install python3-virtualenv  or pip3 install virtualenv
2) source env/bin/activate
3) pip install tensorflow
4) pip install scikit-learn

Then you must first, run the training process launching:
5) python3 nn_train.py

And finally perform the prediction running:
6) python3 nn_prediction.py

The final output is the most suitable aggregation function (OWA or ARTM) with their respective weights (w1,w2 or w3)
