from data_reader import load_data
from preprocessing import fill_missing_numerical_data, get_normalizer_from_data, normalize_numerical_data, handle_categorical_data

 
# Import data
train_data_x, train_data_y = load_data("C:/Users/Fabien/Documents/Covid", "ancien_data.csv")
valid_data_x, valid_data_y = load_data("C:/Users/Fabien/Documents/Covid", "nouveau_data.csv")

# Preprocess data
train_data_x = fill_missing_numerical_data(train_data_x)
valid_data_x = fill_missing_numerical_data(valid_data_x)

normalizer = get_normalizer_from_data(train_data_x)
train_data_x = normalize_numerical_data(normalizer, train_data_x)
valid_data_x = normalize_numerical_data(normalizer, valid_data_x)

train_data_x = handle_categorical_data(train_data_x)
valid_data_x = handle_categorical_data(valid_data_y)

# train data


# validate data

