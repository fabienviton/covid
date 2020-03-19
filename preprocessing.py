import pandas as pd


def numerical_to_categorical(df, drop_numerical=False):
    data_to_convert = {"FC": [0, 50, 120],
                       "GLYCEMIE": [0, 3, 7],
                       "TEMPERATURE": [0, 36, 38],
                       "SATURATION": [0, 100],
                       "OXYGENE": [0, 0],
                       "CETONEMIE": [0, 1],
                       "HEMOCUE": [0, 12],
                       "OH": [0, 0.5],
                       "BLADDER": [0, 10],
                       "PAS": [0, 90, 140],
                       "PAD": [0, 50, 90],
                       "HEURE_ARRIVEE": [0, 7, 19]
                       }


    for key_data in data_to_convert.keys():
        # print(key_data)
        cat_data_df = pd.DataFrame(df, columns=[key_data])

        for i in range(len(data_to_convert[key_data]) - 1):
            inf_boundary = data_to_convert[key_data][i]
            sup_boundary = data_to_convert[key_data][i+1]
            # print("inf, sup boundary", inf_boundary, sup_boundary)
            ids_inf = cat_data_df[key_data] <= sup_boundary
            ids_sup = cat_data_df[key_data] > inf_boundary
            # print("before", df[ids_inf & ids_sup][key_data])
            cat_data_df.loc[ids_inf & ids_sup, key_data] = inf_boundary
            # print("after", df[ids_inf & ids_sup][key_data])
        sup_boundary = data_to_convert[key_data][len(data_to_convert[key_data])-1]
        ids_sup = cat_data_df[key_data] > sup_boundary
        # print("before", df[ids_sup][key_data])
        cat_data_df.loc[ids_sup, key_data] = sup_boundary
        # print("after", df[ids_sup][key_data])

        # generate binary values using get_dummies
        cat_data_df = pd.get_dummies(cat_data_df, columns=[key_data], prefix=[key_data])  # , dummy_na=True)
        print(cat_data_df.shape)
        df = pd.concat([df, cat_data_df.reindex(df.index)], axis=1)
        print(df.shape)
        print(df.columns)

        if drop_numerical:
            df = df.drop(columns=[key_data])

    return df


def fill_missing_numerical_data(df):
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
        # print(col)
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
                        "PREMIER_MOTIF",
                        "JOUR_SEMAINE"]
                        # "SEMAINE_ANNEE",
                        # "HEURE_ARRIVEE"]
                        # "MOTIFS_RECOURS"
    for cat_data in categorical_data:
        # print(cat_data)
        if not (cat_data in df.columns):
            continue
        cat_data_df = pd.DataFrame(df, columns=[cat_data])

        # generate binary values using get_dummies
        dum_df = pd.get_dummies(cat_data_df, columns=[cat_data], prefix=[cat_data])
        # print(dum_df.shape)
        # print(dum_df.columns)
        # merge with main
        # df = df.join(dum_df)
        df = pd.concat([df, dum_df.reindex(df.index)], axis=1)
        # print(df.shape)
        df = df.drop(columns=[cat_data])

        # print(df.shape)

    return df



