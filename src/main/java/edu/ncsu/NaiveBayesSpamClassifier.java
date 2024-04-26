package edu.ncsu;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.util.Random;

public class NaiveBayesSpamClassifier {
	public static void classify() {
		try {
			// Load dataset
			DataSource source = new DataSource("SMSSpamCollection.arff");
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
			Instances newDataWithNominalClass = Filter.useFilter(newDataWithNominalAttributes, filterStringToNominal);

			// Set class index to the newly converted nominal attribute
			newDataWithNominalClass.setClassIndex(newDataWithNominalClass.numAttributes() - 1);

			// Initialize Naive Bayes classifier
			NaiveBayes nb = new NaiveBayes();

			// Train classifier
			nb.buildClassifier(newDataWithNominalClass);

			// Evaluate classifier
			Evaluation eval = new Evaluation(newDataWithNominalClass);
			eval.crossValidateModel(nb, newDataWithNominalClass, 10, new Random(1)); // 10-fold cross-validation
			System.out.println(eval.toSummaryString());

//			// Example prediction
//			Instance instance = new DenseInstance(2);
//			instance.setDataset(newDataWithNominalClass);
//			// Set attributes for the new instance
//			instance.setValue(0, "Your new text message here");
//			// Classify instance
//			double pred = nb.classifyInstance(instance);
//			String prediction = newDataWithNominalClass.classAttribute().value((int) pred);
//			System.out.println("Predicted class: " + prediction);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
