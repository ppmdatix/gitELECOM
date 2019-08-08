# gitELECOM
Internship @ Institut Polytechnique de Paris | Telecom sudParis

## Robustesse des systèmes de détection d’intrusion basés sur l’apprentissage machine


Report [here](https://github.com/ppmdatix/RapportTelecom)

Most of the results presented in the report are gathered in _cGANoDEbergerac_ (as mentioned in the report, GANs is all about names).

MNIST data are included in Keras package, while NSL-KDD data were downloaded from https://www.unb.ca/cic/datasets/nsl.html


To reproduce work (some paths to change):
- run `python3.6 cGANoDEbergerac/main_mnist.py` to reproduce SWAGAN on MNIST
- run `python3.6 cGANoDEbergerac/main_novgan.py` to reproduce NOVGAN on NSLKDD
- run `python3.6 cGANoDEbergerac/main_swagan_trafic_input.py` to reproduce SWAGAN on NSLKDD


**Notes :** 
- GeoGebra files are used to generate TikZ pictures
- many Notebooks are present and were used as drafts