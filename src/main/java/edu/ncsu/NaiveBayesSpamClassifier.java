package edu.ncsu;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class NaiveBayesSpamClassifier {
	public static void classify() {
		try {
			// Load dataset
			DataSource source = new DataSource("SMSSpamCollection.arff");
			Instances data = source.getDataSet();
			if (data.classIndex() == -1) {
				data.setClassIndex(data.numAttributes() - 1);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
