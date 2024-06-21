package com.github.chriswhisker;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.util.Random;

public class SpamClassifier {

	private static final int FOLDS_FOR_EVALUATION = 10;
	// Dataset used for training and testing
	private Instances dataSet;
	private NaiveBayes naiveBayesClassifier;

	/**
	 * Trains the spam classifier using the specified data file.
	 *
	 * @param dataFilePath The file path of the dataset to train the classifier.
	 * @throws Exception If an error occurs during training.
	 */
	public void train(String dataFilePath) throws Exception {
		// Load dataset
		DataSource source = new DataSource(dataFilePath);
		dataSet = source.getDataSet();
		dataSet.setClassIndex(dataSet.numAttributes() - 1);

		preprocessData();

		// Initialize and train Naive Bayes classifier
		naiveBayesClassifier = new NaiveBayes();
		naiveBayesClassifier.buildClassifier(dataSet);

		evaluate();
	}

	/**
	 * Evaluates the trained spam classifier.
	 *
	 * @throws Exception If an error occurs during evaluation.
	 */
	private void evaluate() throws Exception {
		Evaluation evaluation = new Evaluation(dataSet);
		evaluation.crossValidateModel(naiveBayesClassifier, dataSet, FOLDS_FOR_EVALUATION, new Random(1));
		System.out.println(evaluation.toSummaryString());
	}

	/**
	 * Preprocesses the data by converting string attributes to word vectors
	 * and numeric attributes to nominal.
	 *
	 * @throws Exception If an error occurs during preprocessing.
	 */
	private void preprocessData() throws Exception {
		// Convert string attribute to word vectors
		StringToWordVector stringToWordVectorFilter = new StringToWordVector();
		stringToWordVectorFilter.setInputFormat(dataSet);
		Instances dataWithWordVectors = Filter.useFilter(dataSet, stringToWordVectorFilter);

		// Convert numeric attributes to nominal
		NumericToNominal numericToNominalFilter = new NumericToNominal();
		numericToNominalFilter.setInputFormat(dataWithWordVectors);
		Instances dataWithNominalAttributes = Filter.useFilter(dataWithWordVectors, numericToNominalFilter);

		// Convert class attribute to nominal
		StringToNominal stringToNominalFilter = new StringToNominal();
		stringToNominalFilter.setAttributeRange("last");
		stringToNominalFilter.setInputFormat(dataWithNominalAttributes);
		Instances preprocessedData = Filter.useFilter(dataWithNominalAttributes, stringToNominalFilter);

		// Set class index to the newly converted nominal attribute
		preprocessedData.setClassIndex(preprocessedData.numAttributes() - 1);

		dataSet = preprocessedData;
	}

	/**
	 * Tests the trained spam classifier with a given text message.
	 *
	 * @param textMessage The text message to classify.
	 * @throws Exception If an error occurs during classification.
	 */
	public void test(String textMessage) throws Exception {
		System.out.println("Predicting class for message: \"" + textMessage + "\"");

		// Create a new instance and process it
		Instance instance = new DenseInstance(dataSet.numAttributes());
		instance.setDataset(dataSet);
		preprocessInstance(textMessage, instance);

		// Classify instance
		double[] predictionDistribution = naiveBayesClassifier.distributionForInstance(instance);
		double pred = naiveBayesClassifier.classifyInstance(instance);
		String prediction = dataSet.classAttribute().value((int) pred);

		// Print confidence level for each class
		System.out.println("Confidence level for spam: " + String.format("%.2f", predictionDistribution[1] * 100) + "%");
		System.out.println("Confidence level for ham: " + String.format("%.2f", predictionDistribution[0] * 100) + "%");

		// Print prediction
		System.out.println("Predicted class: " + (prediction.equals("1") ? "spam" : "ham"));
	}

	/**
	 * Preprocesses a single text message instance before classification.
	 *
	 * @param textMessage The text message to preprocess.
	 * @param instance The instance to be preprocessed.
	 * @throws Exception If an error occurs during preprocessing.
	 */
	private void preprocessInstance(String textMessage, Instance instance) throws Exception {
		if (dataSet == null) {
			throw new IllegalStateException("Dataset is not initialized.");
		}

		// Split the given text message into words
		String[] words = textMessage.split("\\s+");

		// Set attributes for the new instance based on the example text message
		for (String word : words) {
			// Check if the word exists as an attribute in the dataset
			if (dataSet.attribute(word) != null) {
				Attribute attribute = dataSet.attribute(word);
				int attributeIndex = attribute.index();
				instance.setValue(attributeIndex, 1);
			}
		}

		// Create a new Instances object containing only the instance to be tested
		Instances singleInstanceDataSet = new Instances(dataSet, 0);
		singleInstanceDataSet.add(instance);

		// Preprocess the single instance using the same filter as the training data
		StringToWordVector instanceFilter = new StringToWordVector();
		instanceFilter.setInputFormat(dataSet);
		Instances preprocessedSingleInstanceDataSet = Filter.useFilter(singleInstanceDataSet, instanceFilter);

		// Check if preprocessedSingleInstanceDataSet is not empty
		if (preprocessedSingleInstanceDataSet.numInstances() > 0) {
			// Access the first instance
			instance = preprocessedSingleInstanceDataSet.firstInstance();
			// Copy attribute values from the preprocessed instance to the new instance
			for (int i = 0; i < instance.numAttributes(); i++) {
				if (!instance.isMissing(i)) {
					instance.setValue(i, instance.value(i));
				}
			}
		} else {
			throw new Exception("Preprocessed instance dataset is empty.");
		}
	}

}
