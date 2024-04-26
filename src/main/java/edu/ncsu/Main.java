package edu.ncsu;

public class Main {
	public static void main(String[] args) {
		NaiveBayesSpamClassifier classifier = new NaiveBayesSpamClassifier();
		classifier.trainClassifier("SMSSpamCollection.arff");
		classifier.evaluateClassifier();
		classifier.testModel("what the fuck dude");
	}
}