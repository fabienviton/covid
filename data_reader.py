import os
import pandas as pd
from collections import Counter


def load_data(data_dir, data_file):
    df = pd.read_csv(os.path.join(data_dir, data_file))
    y = df[["CODE_DEST", "SERVICE", "SPECIALITE", "DESTINATION", "HOSPITALISATION"]]
    # "MOTIFS_RECOURS"
    x = df[["CODE_ARRIVEE", "CODE_MOYEN", "SEXE", "ACCOMP", "ATTENTE", "CIRCONSTANCES", "FAMILLE", "CIMU", "FC", "DOULEUR", "GLYCEMIE", "TEMPERATURE", "SATURATION", "OXYGENE", "CETONEMIE", "HEMOCUE", "OH", "BLADDER", "AGE", "JOUR_SEMAINE", "SEMAINE_ANNEE", "HEURE_ARRIVEE", "PAS", "PAD", "PREMIER_MOTIF"]]

    # for col in x:
    #     print(col)
    #     test = df[col].to_numpy()
    #     print(test)
    #     counter = Counter(test)
    #     print(counter)
    return x, y


