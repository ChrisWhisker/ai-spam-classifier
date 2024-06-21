# AI Spam Classifier

This Java program demonstrates an implementation of supervised learning for classifying text messages as spam or ham (non-spam) using the Naive Bayes algorithm. The program utilizes the Weka library, a popular open-source machine learning library for Java, to implement the Naive Bayes spam classifier, which is capable of training on a dataset, evaluating its performance, and classifying new text messages as either spam or ham. The purpose of this application is to showcase how machine learning algorithms can be employed to automate tasks, such as email filtering, by training a model on labeled data and leveraging it to make predictions on new, unseen data.

## The Algorithm
The Naive Bayes algorithm is a simple and effective machine learning technique based on Bayes' theorem, which describes the probability of a hypothesis given some observed evidence. In the context of spam classification, Naive Bayes calculates the probability of a message being spam or ham based on the presence or absence of certain words or features. Despite its simplicity, Naive Bayes has proven to be a robust and efficient algorithm for text classification tasks.



## Weka
Weka is a collection of machine learning algorithms for data mining tasks. It contains tools for data preparation, classification, regression, clustering, association rule mining, and more. Weka provides a flexible and extensible framework for data analysis and modeling, making it a popular choice for machine learning tasks.

## How it works

* **Training**: The ability to train a Naive Bayes model on a labeled dataset, enabling the model to learn patterns and relationships between features.
* **Evaluation**: The capability to evaluate the performance of the trained model on a test dataset, providing insights into its accuracy and reliability.
* **Classification**: The ability to classify new, unseen text messages as either spam or ham, leveraging the trained model to make predictions.

## Usage

**Prerequisite**: To run this code, you need to include the `weka.jar` file (provided in the `lib` directory) as a project library in your chosen IDE.

To use the program, simply compile and run the Java code. The program will prompt you to input a text message, which will then be classified as spam or ham based on the trained model.
