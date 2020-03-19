import os
import pandas as pd
from collections import Counter
import numpy as np


def load_data(data_dir, data_file):
    df = pd.read_csv(os.path.join(data_dir, data_file),
                     dtype={'OH': np.float64,
                            'OXYGENE': np.float64,
                            "CODE_ARRIVEE": np.float64,
                            # "MOTIFS_RECOURS": str,
                            "CIRCONSTANCES": np.float64})
    print("CSV datafile:", data_file)

    # "DESTINATION"
    # "HOSPITALISATION"
    y = df[["CODE_DEST", "SERVICE", "SPECIALITE"]]
    if df.columns.contains("HOSPITALISATION"):
        y[["HOSPITALISATION"]] = df[["HOSPITALISATION"]]
    else:
        dum_df = df[["realite"]]
        y[["HOSPITALISATION"]] = (dum_df[["realite"]] == "HOSPITALISATION")
        # print(y[["HOSPITALISATION"]].head(10))

    # "MOTIFS_RECOURS"
    # "JOUR_SEMAINE", "SEMAINE_ANNEE", "HEURE_ARRIVEE",

    x = df[["CODE_ARRIVEE",
            "CODE_MOYEN",
            "SEXE",
            "ACCOMP",
            "ATTENTE",
            "CIRCONSTANCES",
            "FAMILLE",
            "CIMU",
            "FC",
            "DOULEUR",
            "GLYCEMIE",
            "TEMPERATURE",
            "SATURATION",
            "OXYGENE",
            "CETONEMIE",
            "HEMOCUE",
            "OH",
            "BLADDER",
            ]]
    if df.columns.contains("PREMIER_MOTIF"):
        x[["PREMIER_MOTIF"]] = df[["PREMIER_MOTIF"]]
    if df.columns.contains("PAS"):
        x[["PAS"]] = df[["PAS"]]
    if df.columns.contains("PAD"):
        x[["PAD"]] = df[["PAD"]]
    if df.columns.contains("HEURE_ARRIVEE"):
        x[["HEURE_ARRIVEE"]] = df[["HEURE_ARRIVEE"]]
    if df.columns.contains("PREMIER_MOTIF"):
        x[["JOUR_SEMAINE"]] = df[["JOUR_SEMAINE"]]
    if df.columns.contains("AGE"):
        x[["AGE"]] = df[["AGE"]]
    # for col in x:
    #     print(col)
    #     test = df[col].to_numpy()
    #     print(test)
    #     counter = Counter(test)
    #     print(counter)
    return x, y


