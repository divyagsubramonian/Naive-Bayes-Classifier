import sys
import pandas as pd
import numpy as np
import math

def main():
    # Turning the arguments (filenames) into dataframes
    if len(sys.argv) != 3:
        print("Expected: python3 NaiveBayesClassifier.py <train_data_path> <validation_data_path>")
        sys.exit(1)

    train_path = sys.argv[1]
    validation_path = sys.argv[2]

    try:
        train_data = pd.read_csv(train_path)
        validation_data = pd.read_csv(validation_path)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        sys.exit(1)

    # One-hot encoding "home_wl_pre5" and "away_wl_pre5" columns
    train_data = one_hot_encoding(train_data, "home_wl_pre5")
    train_data = one_hot_encoding(train_data, "away_wl_pre5")
    validation_data = one_hot_encoding(validation_data, "home_wl_pre5")
    validation_data = one_hot_encoding(validation_data, "away_wl_pre5")

    # Save team names from validation dataset
    team_names_validation = validation_data["team_abbreviation_home"].to_list()
    
    # Dataframe with numerical columns
    train_dataset = train_data.drop(columns=["team_abbreviation_home", "team_abbreviation_away", "season_type"])
    validation_dataset = validation_data.drop(columns=["team_abbreviation_home", "team_abbreviation_away", "season_type"])

    # Extracting the rows where the home team lost (0) and won (1)
    classes = separate_by_class(train_data)
    df_class0 = pd.DataFrame(classes[0])
    df_class1 = pd.DataFrame(classes[1])
    
    # Calculate the team probabilities (for the "team_abbreviation_home" column) for each class
    team_probabilities_class0 = calculate_probabilities_teams(df_class0, "team_abbreviation_home")
    team_probabilities_class1 = calculate_probabilities_teams(df_class1, "team_abbreviation_home")

    summarize_class0, summarize_class1 = summarize_by_class(train_dataset)
    probabilities_class0, probabilities_class1 = calculate_probabilities(summarize_class0, summarize_class1, team_probabilities_class0, team_probabilities_class1, team_names_validation, validation_dataset)

    for i in range(validation_dataset.shape[0]):
        if probabilities_class0[i] > probabilities_class1[i]:
            print("0")
        else:
            print("1")

# Turning the W/L string columns into numerical values
def one_hot_encoding(data, col):
    for i in range(len(data[col])):
        score = 0
        current_record = data.loc[i, col]
        
        for j in range(len(current_record)):
            if current_record[j] == "W":
                score += 1
        
        score /= 5.0
        data.loc[i, col] = score
    
    return data

# Calculating the team probabilities given their class
def calculate_probabilities_teams(data, col):
    result = data[col].value_counts(normalize = True)
    return result.to_dict()

# Separating the data by their class (0 - lose, 1 - win)
def separate_by_class(data):
    groups_dictionary = dict()
    for i in range(data.shape[0] - 1):
        row_label = data.loc[i + 1, "label"]
        if row_label not in groups_dictionary.keys():
            groups_dictionary[row_label] = []
        groups_dictionary[row_label].append(data.loc[i + 1])

    return groups_dictionary

# Finding the mean and standard deviation of a list of numbers
def mean_and_stddev(numbers):
    mean = np.mean(numbers)
    std_dev = np.std(numbers)
    return mean, std_dev

# Finding the mean and standard deviation of each column in a dataframe
def summarize(data):
    summaries = []
    for col in data.columns:
        summary = mean_and_stddev(data[col])
        summaries.append(summary)

    del summaries[0]
    return summaries

# Summarizing (finding the mean and standard deviation of the columns) by class
def summarize_by_class(data):    
    classes = separate_by_class(data)
    df_class0 = pd.DataFrame(classes[0])
    df_class1 = pd.DataFrame(classes[1])
    summarize_class0 = summarize(df_class0)
    summarize_class1 = summarize(df_class1)
    
    return summarize_class0, summarize_class1

# Calculating the Gaussian probability density function
def calculate_gaussian(x, mean, std_dev):
    gaussian_pdf = (1 / (math.sqrt(2 * math.pi) * std_dev)) * (math.exp(-((x - mean) ** 2 / (2 * std_dev ** 2))))
    return gaussian_pdf

# Calculating the probabilities for each class
def calculate_probabilities(summarize_class0, summarize_class1, team_probabilities_class0, team_probabilities_class1, team_names_validation, data):
    probabilities_class0 = []
    probabilities_class1 = []

    for team in team_names_validation:
        probabilities_class0.append(team_probabilities_class0[team])
        probabilities_class1.append(team_probabilities_class1[team])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            x = data.iloc[i, j]
            mean_class0 = summarize_class0[j][0]
            std_class0 = summarize_class0[j][1]
            mean_class1 = summarize_class1[j][0]
            std_class1 = summarize_class1[j][1]

            if (std_class0 != 0) and (std_class1 != 0):
                probabilities_class0[i] *= calculate_gaussian(x, mean_class0, std_class0)
                probabilities_class1[i] *= calculate_gaussian(x, mean_class1, std_class1)

    return probabilities_class0, probabilities_class1

if __name__ == "__main__":
    main()