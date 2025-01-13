# Naive Bayes Classifier

The architecture of my Naive Bayes Classifier can be broken down into two main components: fitting the model and computing the probabilities. Fitting the model involved calculating prior probabilities of each class and conditional probabilities for each feature. The probabilities were computed using the Gaussian probability density function and Bayes’ theorem.

My code first takes in two arguments: the training data filename and the validation data filename. Using these arguments, I transform the CSV files into pandas dataframes to easily manipulate the columns and their values. After completing the preprocessing steps, which involves one-hot encoding the categorical variable columns and trimming the data frames to only contain numerical variable columns, I separate the training data by class (where class 0 indicates that the team lost and class 1 indicates that the team won). Following this, I calculate the probability that a certain team wins given their class and find the mean and standard deviation of the values in each column. Finally, I calculate the predictions for each team in the validation dataset by finding the probabilities of each team winning and losing and choosing the class with the higher probability.
one_hot_encoding: This function turns the W/L strings (from the “home_wl_pre5” and “away_wl_pre5”  columns into numerical values. Essentially, it weighs the wins as ones and the losses as zeros and returns the average score.

`calculate_probabilities_teams`: This function calculates the probability that a unique value (a given team) occurs in a given class. The function returns the probability of the provided list of teams winning given their class (based on the training data).

`separate_by_class`: This function separates the data by their class, where class 0 contains the teams that lost and class 1 contains the teams that won. A dictionary is returned, and this dictionary is later separated into two dataframes.

`mean_and_stddev`: This function calculates the mean and standard deviation of a list of numbers. This function serves as a helper function for figuring out the mean and standard deviation of the values in each column.

`summarize`: This function calculates the mean and standard deviation of all the columns in a given dataframe. It returns a list of all the tuples with the derived means and standard deviations.

`summarize_by_class`: This function summarizes (finds the mean and standard deviation of every column) by class, where the two classes are class 0 and class 1.

`calculate_gaussian`: This function calculates the Gaussian probability density function given x (the specific element), the column’s mean, and the column’s standard deviation.

<img width="238" alt="Screenshot 2025-01-12 at 4 19 32 PM" src="https://github.com/user-attachments/assets/061a5323-f655-4c4d-9d0f-df140ae1bd49" />

x = element, μ = column’s mean, σ = column’s standard deviation

`calculate_probabilities`: This function calculates the probabilities for each team in the validation dataset. It finds the probabilities of a given team belonging in class 0 and class 1 and returns the lists (for each class) of probabilities for each row. The probabilities are found by calculating the Gaussian probability for each element (using the element’s corresponding mean and standard deviation).

____

A Naive Bayes Classifier is a probabilistic model based on Bayes’ Theorem, used to classify data into different categories. Bayes' Theorem provides a way to calculate the probability of a class given a particular observation, expressed as:

<img width="246" alt="Screenshot 2025-01-12 at 4 20 41 PM" src="https://github.com/user-attachments/assets/ec5c13ad-13d5-49de-a962-6803de8c3964" />

The model is trained by calculating two sets of probabilities: prior probabilities (the probability of each class in the training set) and conditional probabilities (the probability of each feature value given a particular class). Since Naive Bayes assumes that all features are conditionally independent, we can break down P(B|A) as the product of individual feature probabilities. 

To train my Naive Bayes Classifier, I first found the frequency of the categorical feature values in each class, which are essentially the probabilities of the categorical features. To find the numerical feature probabilities, I first saved all the numerical variable columns in a dataframe and found all the probabilities for each element by inputting the element and its corresponding mean and standard deviation (for both class 0 and class 1) into the Gaussian probability density function. To find the probability for a given team (for each class), I multiplied the categorical variable probabilities with Gaussian probabilities for each element in a specific row together. I calculated two probabilities for each row (team) in the validation dataset, once for class 0 and another time for class 1.

