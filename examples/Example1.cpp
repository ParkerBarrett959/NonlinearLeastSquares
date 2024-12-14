/**
 * A Linear Regression example
 *
 * Problem statement: Given the table of square footage and rent data below,
 * train a linear regression model on the data and make predictions of rent
 * price given an apartment's square footage.
 *
 * Training Data
 * Square Footage = [500, 750, 1000, 1250, 1500, 1750]
 * Rent = [1000, 1300, 2200, 2400, 3150, 3400]
 *
 * Predictions:
 * Square Footage = 1100, Rent = ?
 * Square Footage = 1300, Rent = ?
 * Square Footage = 2000, Rent = ?
 */
#include "SupervisedLearning/Regression/LinearRegression.h"
#include <iostream>

int main() {
  // Create a simple linear regression
  LinearRegression lr;

  // Create a matrux of training data
  Eigen::Matrix<double, 6, 2> trainingData;
  trainingData << 500, 1000, 750, 1300, 1000, 2200, 1250, 2400, 1500, 3150,
      1750, 3400;

  // Add the data to the model and train
  lr.addTrainingData(trainingData);
  bool solveSuccess = lr.solveRegression();
  if (!solveSuccess) {
    std::cout << "Failed to solve regression. Terminating..." << std::endl;
    return 1;
  }

  // Print Model Weights
  Eigen::VectorXd predictorWeights = lr.getPredictorWeights();
  std::cout << "Linear Regression Predictor Weights:" << std::endl;
  for (int i = 0; i < predictorWeights.size(); i++) {
    std::cout << "    i = " << i << ", weight = " << predictorWeights(i)
              << std::endl;
  }

  // Create vector of square footage to make predictions on
  Eigen::Matrix<double, 3, 1> predictionValues;
  predictionValues << 1100, 1300, 2000;

  // Predict the rent
  auto maybePredictionResult = lr.predictBatch(predictionValues);
  std::cout << "Linear Regression Predictions:" << std::endl;
  if (maybePredictionResult.has_value()) {
    Eigen::VectorXd predictionResults = maybePredictionResult.value();
    for (int i = 0; i < predictionResults.size(); i++) {
      std::cout << "    (x,y) = (" << predictionValues(i) << ", "
                << predictionResults(i) << ")" << std::endl;
    }
  } else {
    std::cout << "Failed to make predictions. Terminating..." << std::endl;
    return 1;
  }
  return 0;
}