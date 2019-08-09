from load_data.load_arrhythmia import load_arrhythmia
from loadingCGAN.cgan import Cgan
from time import time
from learning import learning
from evaluation.evaluation import evaluate
from sklearn.ensemble.forest import RandomForestClassifier as RFC

x_train, x_test, y_train, y_test, x_train_cv, y_train_cv = load_arrhythmia()

rfc = RFC()
rfc.fit(x_train, y_train)
print(evaluate(y_test,rfc.predict(x_test)))

data_dim = x_train.shape[1]
latent_dim = 3
activation = "tanh"
noise = "normal"
number_of_gans = 2


cgans = [Cgan(data_dim=data_dim, latent_dim=latent_dim,
                  spectral_normalisation=False,
                  weight_clipping=False, verbose=True,
                  activation=activation,
                  noise=noise) for _ in range(number_of_gans)]

start = time()
cgans = learning(cgans=cgans, x=x_train, y=y_train, x_cv=x_train_cv, y_cv=y_train_cv,
                    number_of_gans=number_of_gans,
                    epochs=50, switches=5, print_mode=False)
end = time()
duration = end - start
cgan = cgans[0]
y_pred = cgan.predict(x_test)
print(cgan.predict_proba(x_test))
print(evaluate(y_test, y_pred))