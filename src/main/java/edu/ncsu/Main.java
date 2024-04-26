package edu.ncsu;

public class Main {
	public static void main(String[] args) throws Exception {
		SpamClassifier classifier = new SpamClassifier();
		classifier.trainClassifier("SMSSpamCollection.arff");

		String testTextMsg = "URGENT We are trying to contact you Last weekends draw shows u have won a £1000 prize GUARANTEED Call 09064017295 Claim code K52 Valid 12hrs 150p pm";
		classifier.testModel(testTextMsg);
	}
}