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

public class NaiveBayesSpamClassifier {

	private Instances newDataWithNominalClass;
	private NaiveBayes nb;

	public void trainClassifier(String dataFilePath) {
		try {
			// Load dataset
			DataSource source = new DataSource(dataFilePath);
			Instances data = source.getDataSet();
			if (data.classIndex() == -1) {
				data.setClassIndex(data.numAttributes() - 1);
			}

			// Convert string attribute to word vectors
			StringToWordVector filter = new StringToWordVector();
			filter.setInputFormat(data);
			Instances newData = Filter.useFilter(data, filter);

			// Convert numeric attributes to nominal
			NumericToNominal filterNumericToNominal = new NumericToNominal();
			filterNumericToNominal.setInputFormat(newData);
			Instances newDataWithNominalAttributes = Filter.useFilter(newData, filterNumericToNominal);

			// Convert class attribute to nominal
			StringToNominal filterStringToNominal = new StringToNominal();
			filterStringToNominal.setAttributeRange("last");
			filterStringToNominal.setInputFormat(newDataWithNominalAttributes);
			newDataWithNominalClass = Filter.useFilter(newDataWithNominalAttributes, filterStringToNominal);

			// Set class index to the newly converted nominal attribute
			newDataWithNominalClass.setClassIndex(newDataWithNominalClass.numAttributes() - 1);

			// Initialize Naive Bayes classifier
			nb = new NaiveBayes();

			// Train classifier
			nb.buildClassifier(newDataWithNominalClass);

			// Evaluate classifier
			Evaluation eval = new Evaluation(newDataWithNominalClass);
			eval.crossValidateModel(nb, newDataWithNominalClass, 10, new Random(1)); // 10-fold cross-validation
			System.out.println(eval.toSummaryString());

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void evaluateClassifier() {
		try {
			// Evaluate classifier
			Evaluation eval = new Evaluation(newDataWithNominalClass);
			eval.crossValidateModel(nb, newDataWithNominalClass, 10, new Random(1)); // 10-fold cross-validation
			System.out.println(eval.toSummaryString());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}


	public int testModel(String textMessage) {
		try {

			// Create a new instance
			Instance instance = new DenseInstance(newDataWithNominalClass.numAttributes());
			instance.setDataset(newDataWithNominalClass);

			// Split the given text message into words
			String[] words = textMessage.split("\\s+");

			// Set attributes for the new instance based on the example text message
			for (String word : words) {
				// Find the index of the attribute corresponding to the word
				Attribute attribute = newDataWithNominalClass.attribute(word.toLowerCase());
				if (attribute != null) {
					int attributeIndex = attribute.index();
					// Set the value for the attribute
					instance.setValue(attributeIndex, 1); // Assuming word occurrence is binary
				}
			}

			// Create a new Instances object containing only the instance to be tested
			Instances singleInstanceDataset = new Instances(newDataWithNominalClass, 0);
			singleInstanceDataset.add(instance);

			// Preprocess the single instance using the same filter as the training data
			StringToWordVector filterInstance = new StringToWordVector();
			filterInstance.setInputFormat(singleInstanceDataset);
			Instances preprocessedSingleInstanceDataset = Filter.useFilter(singleInstanceDataset, filterInstance);

			// Get the preprocessed instance
			Instance preprocessedInstance = preprocessedSingleInstanceDataset.get(0);

			// Classify instance
			double pred = nb.classifyInstance(preprocessedInstance);
			String prediction = newDataWithNominalClass.classAttribute().value((int) pred);
			System.out.println("Predicted class: " + (Objects.equals(prediction, "1") ? "spam" : "ham"));

			return Integer.parseInt(prediction);
		} catch (Exception e) {
			e.printStackTrace();
			return 0;
		}
	}
}
