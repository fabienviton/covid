import pandas as pd


def fill_missing_numerical_data(df):
    print("before", df["FC"].to_numpy())
    default_values = {"FC": 80,
                      "DOULEUR": 0,
                      "GLYCEMIE": 5.0,
                      "TEMPERATURE": 37.4,
                      "SATURATION": 100,
                      "OXYGENE": 0,
                      "CETONEMIE": 0,
                      "HEMOCUE": 12.0,
                      "OH": 0,
                      "BLADDER": 0,
                      "PAS": 120,
                      "PAD": 80}
    df = df.fillna(value=default_values)
    return df


def get_normalizer_from_data(df):
    norm = {}
    numerical_data = ["FC", "DOULEUR", "GLYCEMIE", "TEMPERATURE", "SATURATION", "OXYGENE", "CETONEMIE", "HEMOCUE", "OH", "BLADDER", "PAS", "PAD", "AGE"]
    for col in df:
        if col in numerical_data:
            print(col)
            np_col = df[col].to_numpy()
            print(np_col)
            print("Moyenne", np_col.mean())
            print("Std", np_col.std())
            norm[col] = {'mean': np_col.mean(), 'std': np_col.std()}
    return norm


def normalize_numerical_data(normalizer, df):
    for col in normalizer:
        df[col] = (df[col] - normalizer[col]['mean']) / normalizer[col]['std']
    return df


def handle_categorical_data(df):
    # TODO: "MOTIFS_RECOURS" à gérer car constitué de liste.
    # Si "PREMIER_MOTIF", il amène 110 catégories ce qui pourrait pourrir le modèle ? A voir
    categorical_data = ["CODE_ARRIVEE",
                        "CODE_MOYEN",
                        "SEXE",
                        "ACCOMP",
                        "ATTENTE",
                        "CIRCONSTANCES",
                        "FAMILLE",
                        "CIMU",
                        "PREMIER_MOTIF"]
                        # "JOUR_SEMAINE",
                        # "SEMAINE_ANNEE",
                        # "HEURE_ARRIVEE"]
                        # "MOTIFS_RECOURS"
    for cat_data in categorical_data:
        print(cat_data)
        if not (cat_data in df.columns):
            continue
        cat_data_df = pd.DataFrame(df, columns=[cat_data])

        # generate binary values using get_dummies
        dum_df = pd.get_dummies(cat_data_df, columns=[cat_data], prefix=[cat_data])
        print(dum_df.shape)
        print(dum_df.columns)
        # merge with main
        # df = df.join(dum_df)
        df = pd.concat([df, dum_df.reindex(df.index)], axis=1)
        print(df.shape)
        df = df.drop(columns=[cat_data])

        print(df.shape)

    return df



