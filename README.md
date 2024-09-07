# MCLARENcpp
MCLARENcpp MaChine LeArning pREdictioN of Cell Penetrating Peptide and their uptake efficiency


Here is the translation in English:

Our source code is titled MCLAREN.py, and in it, you will find how the descriptor calculations were done and how the model was trained. Additionally, the file contains three main functions:

training_matrix(positives_path, negatives_path, save_path)
training_model(training_matrix_path, save_path)
test_model(test_matrix_path, training_matrix_path, model_path, save_name)

The first function, training_matrix, takes two files, positives_path and negatives_path (in CSV or FASTA format), calculates the descriptors, and saves them into a single file with a user-defined name (save_name).

Note: You need to run the training_matrix function for both the test and training data.

The second function, training_model, requires the path to the training dataset you generated with the previous function as a parameter, and you will define a path to save the model.

Finally, test_model takes the test dataset, the training dataset (to ensure the test data was not part of the training), the model path generated from the previous function, and the path where you want to save the results.
