package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.TreeMap;

import tud.ke.ml.project.framework.classifier.ANearestNeighbor;
import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but
 * probably not set)
 * 
 * @author cwirth
 *
 */
public class NearestNeighbor extends ANearestNeighbor {

	protected double[] scaling;
	protected double[] translation;
	private List<List<Object>> trainingsData;

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> votes = null;
		if (!isInverseWeighting()) {
			// Get all unweighted Votes
			votes = getUnweightedVotes(subset);
		} else {
			// Get all weighted Votes
			votes = getWeightedVotes(subset);
		}

		// Return winner class
		return getWinner(votes);
	}

	@Override
	protected void learnModel(List<List<Object>> traindata) {
		// Save the training data
		trainingsData = traindata;
	}

	@Override
	protected Map<Object, Double> getUnweightedVotes(
			List<Pair<List<Object>, Double>> subset) {

		// Map for the unweighted votes results
		Map<Object, Double> result = new HashMap<Object, Double>();

		// Iterate over all instances in the subset
		for (Pair<List<Object>, Double> instance : subset) {

			// Get the class attribut
			Object classAttribut = instance.getA().get(getClassAttribute());

			// Count the votes
			double number = 0.0;
			if (!result.containsKey(classAttribut)) {
				number = 1.0;
			} else {
				number = result.get(classAttribut) + 1.0;
				result.remove(classAttribut);
			}

			// Update the Result
			result.put(classAttribut, number);
		}
		return result;
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		// Map for the weighted votes results
		Map<Object, Double> result = new HashMap<Object, Double>();

		// Iterate over all instances in the subset
		for (Pair<List<Object>, Double> instance : subset) {

			// Get the class attribut
			Object classAttribut = instance.getA().get(getClassAttribute());
			// Get the distance
			double distance = Math.pow((double) instance.getB(), -1);

			// New Entry
			double newVote = 0;
			if (result.containsKey(classAttribut)) {
				newVote = result.get(classAttribut) + distance;
			} else {
				newVote = distance;
			}

			// Remove old entry
			result.remove(classAttribut);

			// Update the Result
			result.put(classAttribut, newVote);
		}
		return result;
	}

	@Override
	protected Object getWinner(Map<Object, Double> votesFor) {
		// Data of the winner
		double max = 0;
		Object winnerClass = null;

		// get class distribution in order to decide tie breakers
		Map<Object, Integer> classDistr = getClassDistribution();

		// Search the winner
		for (Entry entry : votesFor.entrySet()) {
			// is it a tie breaker but the quantity of the current class is
			// higher -> choose it
			if ((double) entry.getValue() == max
					&& classDistr.get(winnerClass) < classDistr.get(entry
							.getKey())) {
				max = (double) entry.getValue();
				winnerClass = entry.getKey();
			}
			// normal winner check
			else if ((double) entry.getValue() > max) {
				max = (double) entry.getValue();
				winnerClass = entry.getKey();
			}
		}
		return winnerClass;
	}

	/**
	 * This method is used to get the class distribution
	 */
	private Map<Object, Integer> getClassDistribution() {
		Map<Object, Integer> classDistribution = new HashMap<Object, Integer>();
		for (List<Object> instance : trainingsData) {
			Object classAttr = instance.get(getClassAttribute());
			if (classDistribution.keySet().contains(classAttr)) {
				classDistribution.put(classAttr,
						classDistribution.get(classAttr) + 1);
			} else {
				classDistribution.put(classAttr, 1);
			}
		}
		return classDistribution;
	}

	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> testdata) {
		// List to save the result
		List<Pair<List<Object>, Double>> result = new ArrayList<Pair<List<Object>, Double>>();
		// get translation and scaling vector and save it
		double[][] normalisationArray = normalizationScaling();
		scaling = normalisationArray[0];
		translation = normalisationArray[1];
		// normalise testdata
		List<Object> normalisedTestdata = new ArrayList<Object>(), normalisedTrainingdata;
		for (int i = 0; i < testdata.size(); i++) {
			Object curCondition = testdata.get(i);
			// check whether it is a nominal attribute
			if (curCondition instanceof String) {
				normalisedTestdata.add(curCondition);
			} else {
				if (scaling[i] == 0.0) {
					normalisedTestdata.add((double) curCondition);
				} else {
					normalisedTestdata
							.add(((double) curCondition - translation[i])
									/ scaling[i]);

				}
			}
		}

		// Calculate the distances and save it in the result list
		for (List<Object> trainingsInstance : trainingsData) {
			normalisedTrainingdata = new ArrayList<Object>();
			for (int i = 0; i < trainingsInstance.size(); i++) {
				Object curCondition = trainingsInstance.get(i);
				// check whether it is a nominal attribute
				if (curCondition instanceof String) {
					normalisedTrainingdata.add(curCondition);
				} else {
					if (scaling[i] == 0.0) {
						normalisedTrainingdata.add((double) curCondition);
					} else {
						normalisedTrainingdata
								.add(((double) curCondition - translation[i])
										/ scaling[i]);
					}
				}
			}
			double distance = 0;
			if (getMetric() == 0) {
				distance = determineManhattanDistance(normalisedTestdata,
						normalisedTrainingdata);
			} else {
				distance = determineEuclideanDistance(normalisedTestdata,
						normalisedTrainingdata);
			}

			Pair<List<Object>, Double> resultEntry = new Pair<List<Object>, Double>(
					normalisedTrainingdata, distance);
			result.add(resultEntry);
		}

		// Get the k nearest Instances
		Comparator<Pair<List<Object>, Double>> comp = new Comparator<Pair<List<Object>, Double>>() {

			@Override
			public int compare(Pair<List<Object>, Double> o1,
					Pair<List<Object>, Double> o2) {
				return o1.getB().compareTo(o2.getB());
			}
		};

		// Sort the list and get the first k instances
		Collections.sort(result, comp);

		int to = Math.min(result.size(), getkNearest());

		// Return the list of the first k instances
		if (to > 1) {
			// if the last kth instance has the same distance than k+1th
			// instance -> increase k so that it is included for voting
			while (result.size() > to
					&& result.get(to - 1).getB() == result.get(to).getB()) {
				to++;
			}
			return result.subList(0, to);
		} else {
			Pair<List<Object>, Double> resultFinal = result.get(0);
			List<Pair<List<Object>, Double>> resultList = new ArrayList<Pair<List<Object>, Double>>();
			resultList.add(resultFinal);
			return resultList;
		}

	}

	@Override
	protected double determineManhattanDistance(List<Object> instance1,
			List<Object> instance2) {
		// Set distance to 0
		double result = 0;

		// Calculate the Distance
		for (int i = 0; i < instance1.size(); i++) {

			// Skip Class Attribut
			if (i == getClassAttribute())
				continue;
			// If it is a String
			if (instance1.get(i) instanceof String) {
				if (!instance1.get(i).equals(instance2.get(i))) {
					result++;
				}

			} else {
				// If it is numeric
				result = result
						+ Math.abs(((double) instance1.get(i) - (double) instance2
								.get(i)));
			}
		}
		return result;
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1,
			List<Object> instance2) {
		// Set distance to 0
		double result = 0;

		// Calculate the Distance
		for (int i = 0; i < instance1.size(); i++) {

			// Skip the class attribut
			if (i == getClassAttribute())
				continue;

			double d = 0.0;
			// If it is a String
			if (instance1.get(i) instanceof String) {
				if (!instance1.get(i).equals(instance2.get(i))) {
					d = 1.0;
				}

			} else {
				// If it is numeric
				d = Math.abs(((double) instance1.get(i) - (double) instance2
						.get(i)));
			}
			result = result + Math.pow(d, 2);
		}
		return Math.sqrt(result);
	}

	@Override
	protected double[][] normalizationScaling() {
		List<Double> minValues = new ArrayList<Double>();
		List<Double> maxValues = new ArrayList<Double>();
		double[][] result = new double[2][trainingsData.get(0).size()];
		// initialise lists with first instance
		for (Object curCondition : trainingsData.get(0)) {

			if (curCondition instanceof String) {
				minValues.add(0.0);
				maxValues.add(1.0);
			} else {
				minValues.add((double) curCondition);
				maxValues.add((double) curCondition);
			}
		}
		if (isNormalizing()) {
			// for each attribute check whether the current instance is the
			// lowest discovered value or the highest discovered value
			for (List<Object> instance : trainingsData) {
				for (int i = 0; i < instance.size(); i++) {
					Object curCondition = instance.get(i);
					// enter normal scaling and translation values for nominal
					// values
					if (curCondition instanceof String) {
						minValues.set(i, 0.0);
						maxValues.set(i, 1.0);
					} else {
						// check whether attribute of current instance is lowest
						// or highest discovered value
						if ((double) curCondition < (double) minValues.get(i)) {
							minValues.set(i, (double) curCondition);
						}
						if ((double) curCondition > (double) maxValues.get(i)) {
							maxValues.set(i, (double) curCondition);
						}
					}
				}
			}
			// calculate scaling and translation value
			for (int i = 0; i < result[0].length; i++) {
				result[0][i] = maxValues.get(i) - minValues.get(i);
				result[1][i] = minValues.get(i);
			}
		} else {
			// normal translation and scaling value if normalisation is not set
			for (int i = 0; i < result[0].length; i++) {
				result[0][i] = 1.0;
				result[1][i] = 0.0;
			}
		}
		return result;
	}

	@Override
	protected String[] getMatrikelNumbers() {
		String[] matrikelNumbers = new String[] { "1945847", "1946134" };
		return matrikelNumbers;
	}

}
