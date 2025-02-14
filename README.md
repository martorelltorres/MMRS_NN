This repo works under Ubuntu 20.04 using Tensorflow and Keras.
Installation procedure:


1) sudo apt install python3-virtualenv  or pip3 install virtualenv
2) virtualenv env
3) source env/bin/activate
4) pip install tensorflow scikit-learn numpy matplotlib pandas

Then you must first, run the training process launching: python3 nn_train.py
And finally perform the prediction running: python3 nn_prediction.py

The final output is the most suitable aggregation function (OWA or ARTM) with their respective weights (w1,w2,w3 or a,b respectively)
