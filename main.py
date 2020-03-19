import importlib
import os
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Masking, Dropout, Conv1D, Conv2D, Reshape, AveragePooling1D, GlobalAveragePooling1D, BatchNormalization, Multiply
from keras.optimizers import Adam

from data_reader import load_data
from preprocessing import fill_missing_numerical_data, get_normalizer_from_data, normalize_numerical_data, handle_categorical_data, numerical_to_categorical
from callbacks import HospitalisationMetrics, CSVLogger, ModelCheckpoint


# Import data. X a 25 columns, Y en a 5.
ancien_data_x, ancien_data_y = load_data("C:/Users/Fabien/Documents/Covid", "ancien_data.csv")
nouveau_data_x, nouveau_data_y = load_data("C:/Users/Fabien/Documents/Covid", "nouveau_data.csv")
prospective_data_x, prospective_data_y = load_data("C:/Users/Fabien/Documents/Covid", "passages_2020-01-17.csv")
prospective_data2_x, prospective_data2_y = load_data("C:/Users/Fabien/Documents/Covid", "passages_2020-03-17 - Cleaned.csv")


# PREPROCESS DATA
# GESTION VARIABLES NUMERIQUES
ancien_data_x = fill_missing_numerical_data(ancien_data_x)
nouveau_data_x = fill_missing_numerical_data(nouveau_data_x)
prospective_data_x = fill_missing_numerical_data(prospective_data_x)
prospective_data2_x = fill_missing_numerical_data(prospective_data2_x)

# drop_numerical=False
# train_data_x = numerical_to_categorical(train_data_x, drop_numerical=drop_numerical)
# valid_data_x = numerical_to_categorical(valid_data_x, drop_numerical=drop_numerical)
# prospective_data_x = numerical_to_categorical(prospective_data_x, drop_numerical=drop_numerical)
# prospective_data2_x = numerical_to_categorical(prospective_data2_x, drop_numerical=drop_numerical)

normalizer = get_normalizer_from_data(ancien_data_x)
ancien_data_x = normalize_numerical_data(normalizer, ancien_data_x)
nouveau_data_x = normalize_numerical_data(normalizer, nouveau_data_x)
prospective_data_x = normalize_numerical_data(normalizer, prospective_data_x)
prospective_data2_x = normalize_numerical_data(normalizer, prospective_data2_x)


# GESTION VARIABLES CATEGORIELLES:
# Passage de 25 à 191 colonnes dans X.

# Regroupement des dataset de train et test avant de gérer le catégoriel (one hot encoding des valeurs).
# print("train_data_x.shape", train_data_x.shape)
# print("valid_data_x.shape", valid_data_x.shape)
# print("test_data_x.shape", test_data_x.shape)
nb_ancien_data_x = ancien_data_x.shape[0]
nb_nouveau_data_x = nouveau_data_x.shape[0]
nb_prosp_data1_x = prospective_data_x.shape[0]
all_data_x = pd.concat([ancien_data_x, nouveau_data_x, prospective_data_x, prospective_data2_x])

# TODO: Transformation des variables ?
# Jour_semaine -> semaine/WE ?
# semaine_annee -> saison ?
# heure_arrivee -> matin/AM/nuit ?

# TODO: gestion MOTIFS_RECOURS

# Transformation en elle meme
all_data_x = handle_categorical_data(all_data_x)

# reséparation des dataset
print("all_data_x.shape", all_data_x.shape)
ancien_data_x = all_data_x[:nb_ancien_data_x]
nouveau_data_x = all_data_x[nb_ancien_data_x:nb_ancien_data_x + nb_nouveau_data_x]
prospective_data_x = all_data_x[nb_ancien_data_x + nb_nouveau_data_x:nb_ancien_data_x + nb_nouveau_data_x + nb_prosp_data1_x]
prospective_data2_x = all_data_x[nb_ancien_data_x + nb_nouveau_data_x + nb_prosp_data1_x:]



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
ancien_data_y = ancien_data_y['HOSPITALISATION']
nouveau_data_y = nouveau_data_y['HOSPITALISATION']
prospective_data_y = prospective_data_y['HOSPITALISATION']
prospective_data2_y = prospective_data2_y['HOSPITALISATION']

X_train_pros, X_test_pros, y_train_pros, y_test_pros = train_test_split(prospective_data2_x, prospective_data2_y, test_size=0.25, random_state=12)



# CONSTRUCTION DU MODELE
# TODO: TESTER DIFFERENTS MODELES :p
# ici modèle MLP simple pour tester le modèle.
data_shape = ancien_data_x.shape[1]
X = Input(shape=(data_shape, ), name='X')
layer_1 = Dense(units=128, activation='relu', name='layer_1')(X)
layer_1 = Dropout(0.3)(layer_1)
# layer_2 = Dense(units=128, activation='relu', name='layer_2')(layer_1)
# layer_2 = Dropout(0.5)(layer_2)
# layer_3 = Dense(units=128, activation='relu', name='layer_3')(layer_2)
# layer_3 = Dropout(0.5)(layer_3)
Y = Dense(units=1, activation='sigmoid', name='final_pred')(layer_1)

model = Model(inputs=X, outputs=Y)
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy')
print(model.summary())


# CREATION DES CALLBACKS POUR LES METRIQUES et SAUVEGARDE DES MODELES
# hospit_metrics = HospitalisationMetrics(train_data_x=train_data_x,
#                                         train_data_y=train_data_y,
#                                         val_data_x=valid_data_x,
#                                         val_data_y=valid_data_y,
#                                         test_data_x=prospective_data_x,
#                                         test_data_y=prospective_data_y)


res_dir = "C:/Users/Fabien/CovidResult"
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# Callback de save des logs
keras_logs = os.path.join(res_dir, 'keras_logs')
if not os.path.exists(keras_logs):
    os.makedirs(keras_logs)
csv_logger = CSVLogger(os.path.join(keras_logs, "1hidden128" + '.csv'),
                       append=True, separator=';')

# Callback de save des modeles
keras_states = os.path.join(res_dir, 'keras_states')
if not os.path.exists(keras_states):
    os.makedirs(keras_states)
saver = ModelCheckpoint(os.path.join(keras_states, "1hidden128" + '.epoch{epoch}.state'), verbose=1, period=1)

# Je me sers du StratifiedKFold pour spliter le dataset retro ancien en 15.
# Chaque quinzieme sera associé à la totalité du dataset train prospectif pour l'entrainement.
nb_split = 50
skf = StratifiedKFold(n_splits=nb_split)
i_split = 0
for _, train_index in skf.split(ancien_data_x, ancien_data_y):
    print("SPLIT NUMERO", str(i_split), "SUR", str(nb_split))
    # X_train = ancien_data_x.loc[train_index]
    # y_train = ancien_data_y.loc[train_index]
    #
    # X_train = pd.concat([X_train, X_train_pros])
    # y_train = pd.concat([y_train, y_train_pros])
    #
    # # shuffle data
    # print("preshuffle", X_train.shape, y_train.shape)
    # col_y = y_train.name
    # all_data_train = pd.concat([X_train, y_train.reindex(X_train.index)], axis=1)
    # print("after concat", all_data_train.shape)
    # all_data_train = all_data_train.sample(frac=1)
    # y_train = all_data_train[col_y]
    # X_train = all_data_train.drop(columns=[col_y])
    # print("postshuffle", X_train.shape, y_train.shape)

    X_train = X_train_pros
    y_train = y_train_pros

    hospit_metrics = HospitalisationMetrics(train_data_x=X_train,
                                            train_data_y=y_train,
                                            train_label="TRAIN mix retro/prospectif",
                                            val_data_x=X_test_pros,
                                            val_data_y=y_test_pros,
                                            valid_label="VAL prospectif",
                                            test_data_x=nouveau_data_x,
                                            test_data_y=nouveau_data_y,
                                            test_label="val retrospectif"
                                            )
    # train retrospective Jan15-Dec18
    # valid retrospective Jan19-Jun19
    # valid prospective Nov19-Dec19
    # prospective Nov19-Mar20
    # ENTRAINEMENT
    model.fit(x=X_train,
              y=y_train,
              epochs=50,
              verbose=2,
              # validation_data=(valid_data_x, valid_data_y),
              callbacks=[hospit_metrics, csv_logger, saver])

