import importlib
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout, Conv1D, Conv2D, Reshape, AveragePooling1D, GlobalAveragePooling1D, BatchNormalization, Multiply

from data_reader import load_data
from preprocessing import fill_missing_numerical_data, get_normalizer_from_data, normalize_numerical_data, handle_categorical_data
from callbacks import HospitalisationMetrics


# Import data. X a 25 columns, Y en a 5.
train_data_x, train_data_y = load_data("C:/Users/Fabien/Documents/Covid", "ancien_data.csv")
valid_data_x, valid_data_y = load_data("C:/Users/Fabien/Documents/Covid", "nouveau_data.csv")
test_data_x, test_data_y = load_data("C:/Users/Fabien/Documents/Covid", "passages_2020-01-17.csv")


# Preprocess data
train_data_x = fill_missing_numerical_data(train_data_x)
valid_data_x = fill_missing_numerical_data(valid_data_x)
test_data_x = fill_missing_numerical_data(test_data_x)

normalizer = get_normalizer_from_data(train_data_x)
train_data_x = normalize_numerical_data(normalizer, train_data_x)
valid_data_x = normalize_numerical_data(normalizer, valid_data_x)
test_data_x = normalize_numerical_data(normalizer, test_data_x)


# GESTION VARIABLES CATEGORIELLES:
# Passage de 25 à 191 colonnes dans X.

# Regroupement des dataset de train et test avant de gérer le catégoriel (one hot encoding des valeurs).
# print("train_data_x.shape", train_data_x.shape)
# print("valid_data_x.shape", valid_data_x.shape)
# print("test_data_x.shape", test_data_x.shape)
nb_train_data_x = train_data_x.shape[0]
nb_valid_data_x = valid_data_x.shape[0]
all_data_x = pd.concat([train_data_x, valid_data_x, test_data_x])

# TODO: Transformation des variables
# Jour_semaine -> semaine/WE ?
# semaine_annee -> saison ?
# heure_arrivee -> matin/AM/nuit ?

# TODO: gestion MOTIFS_RECOURS

# Transformation en elle meme
all_data_x = handle_categorical_data(all_data_x)

# reséparation des dataset
print("all_data_x.shape", all_data_x.shape)
train_data_x = all_data_x[:nb_train_data_x]
valid_data_x = all_data_x[nb_train_data_x:nb_train_data_x+nb_valid_data_x]
test_data_x = all_data_x[nb_train_data_x+nb_valid_data_x:]


# Suppression des données catégorielles dans X: 25 -> 16 colonnes.
# categorical_data = ["CODE_ARRIVEE", "CODE_MOYEN", "SEXE", "ACCOMP", "ATTENTE", "CIRCONSTANCES", "FAMILLE", "CIMU",
#                     "PREMIER_MOTIF"]
# train_data_x = train_data_x.drop(columns=categorical_data)
# valid_data_x = valid_data_x.drop(columns=categorical_data)
# print(train_data_x.shape, train_data_y.shape)
# print(valid_data_x.shape, valid_data_y.shape)
# print(train_data_x.columns)


# GESTION DES VARIABLES RESULTATS
# TODO: Gérer d'autres variables de sorties
# Je ne considère que l'hospitalisation fiale en résultat: Y passe de 5 à une colonne.
train_data_y = train_data_y['HOSPITALISATION']
valid_data_y = valid_data_y['HOSPITALISATION']
test_data_y = test_data_y['HOSPITALISATION']



# CONSTRUCTION DU MODELE
# TODO: TESTER DIFFERENTS MODELES :p
# ici modèle MLP simple pour tester le modèle.
data_shape = train_data_x.shape[1]
X = Input(shape=(data_shape, ), name='X')
layer_1 = Dense(units=64, activation='relu', name='layer_1')(X)
layer_1 = Dropout(0.3)(layer_1)
# layer_2 = Dense(units=32, activation='relu', name='layer_2')(layer_1)
# layer_2 = Dropout(0.5)(layer_2)
Y = Dense(units=1, activation='sigmoid', name='final_pred')(layer_1)

model = Model(inputs=X, outputs=Y)
model.compile(optimizer='Adam', loss='binary_crossentropy')
print(model.summary())


# CREATION DES CALLBACKS POUR LES METRIQUES et SAUVEGARDE DES MODELES
hospit_metrics = HospitalisationMetrics(train_data_x=train_data_x,
                                        train_data_y=train_data_y,
                                        val_data_x=valid_data_x,
                                        val_data_y=valid_data_y,
                                        test_data_x=test_data_x,
                                        test_data_y=test_data_y)
# TODO: Faire un callback pour save le modele à chaque epoch afin de choisr le meilleur à la fin


# ENTRAINEMENT
model.fit(x=train_data_x,
          y=train_data_y,
          epochs=50,
          verbose=2,
          # validation_data=(valid_data_x, valid_data_y),
          callbacks=[hospit_metrics])

