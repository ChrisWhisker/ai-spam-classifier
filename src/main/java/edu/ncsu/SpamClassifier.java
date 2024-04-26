package edu.ncsu;

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

import java.util.Objects;
import java.util.Random;

public class SpamClassifier {

	private Instances dataSet;
	private NaiveBayes naiveBayesClassifier;

	public void trainClassifier(String dataFilePath) {
		try {
			// Load dataset
			DataSource source = new DataSource(dataFilePath);
			Instances rawData = source.getDataSet();
			if (rawData.classIndex() == -1) {
				rawData.setClassIndex(rawData.numAttributes() - 1);
			}

			// Convert string attribute to word vectors
			StringToWordVector stringToWordVectorFilter = new StringToWordVector();
			stringToWordVectorFilter.setInputFormat(rawData);
			Instances dataWithWordVectors = Filter.useFilter(rawData, stringToWordVectorFilter);

			// Convert numeric attributes to nominal
			NumericToNominal numericToNominalFilter = new NumericToNominal();
			numericToNominalFilter.setInputFormat(dataWithWordVectors);
			Instances dataWithNominalAttributes = Filter.useFilter(dataWithWordVectors, numericToNominalFilter);

			// Convert class attribute to nominal
			StringToNominal stringToNominalFilter = new StringToNominal();
			stringToNominalFilter.setAttributeRange("last");
			stringToNominalFilter.setInputFormat(dataWithNominalAttributes);
			dataSet = Filter.useFilter(dataWithNominalAttributes, stringToNominalFilter);

			// Set class index to the newly converted nominal attribute
			dataSet.setClassIndex(dataSet.numAttributes() - 1);

			// Initialize Naive Bayes classifier
			naiveBayesClassifier = new NaiveBayes();

			// Train classifier
			naiveBayesClassifier.buildClassifier(dataSet);

			// Evaluate classifier
			Evaluation evaluation = new Evaluation(dataSet);
			evaluation.crossValidateModel(naiveBayesClassifier, dataSet, 10, new Random(1)); // 10-fold cross-validation
			System.out.println(evaluation.toSummaryString());

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void evaluateClassifier() {
		try {
			// Evaluate classifier
			Evaluation evaluation = new Evaluation(dataSet);
			evaluation.crossValidateModel(naiveBayesClassifier, dataSet, 10, new Random(1)); // 10-fold cross-validation
			System.out.println(evaluation.toSummaryString());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void testModel(String textMessage) {
		try {
			System.out.println("Predicting class for message: \"" + textMessage + "\"");

			// Create a new instance
			Instance instance = new DenseInstance(dataSet.numAttributes());
			instance.setDataset(dataSet);

			// Split the given text message into words
			String[] words = textMessage.split("\\s+");

			// Set attributes for the new instance based on the example text message
			for (String word : words) {
				// Find the index of the attribute corresponding to the word
				Attribute attribute = null;
				for (int i = 0; i < dataSet.numAttributes(); i++) {
					if (dataSet.attribute(i).name().equalsIgnoreCase(word)) {
						attribute = dataSet.attribute(i);
						break;
					}
				}
				if (attribute != null) {
					int attributeIndex = attribute.index();
					// Set the value for the attribute
					instance.setValue(attributeIndex, 1); // Assuming word occurrence is binary
				} else {
					System.out.println("Skipping unknown word: " + word);
				}
			}

			// Create a new Instances object containing only the instance to be tested
			Instances singleInstanceDataSet = new Instances(dataSet, 0);
			singleInstanceDataSet.add(instance);

			// Preprocess the single instance using the same filter as the training data
			StringToWordVector instanceFilter = new StringToWordVector();
			instanceFilter.setInputFormat(singleInstanceDataSet);
			Instances preprocessedSingleInstanceDataSet = Filter.useFilter(singleInstanceDataSet, instanceFilter);

			// Get the preprocessed instance
			Instance preprocessedInstance = preprocessedSingleInstanceDataSet.get(0);

			// Classify instance
			double[] predictionDistribution = naiveBayesClassifier.distributionForInstance(preprocessedInstance);
			double pred = naiveBayesClassifier.classifyInstance(preprocessedInstance);
			String prediction = dataSet.classAttribute().value((int) pred);
			System.out.println("Predicted class: " + (Objects.equals(prediction, "1") ? "spam" : "ham"));

			// Print confidence level for each class
			System.out.println("Confidence level for spam: " + String.format("%.2f", predictionDistribution[1] * 100) + "%");
			System.out.println("Confidence level for ham: " + String.format("%.2f", predictionDistribution[0] * 100) + "%");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
