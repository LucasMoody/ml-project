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
 * This implementation assumes the class attribute is always available (but probably not set)
 * @author cwirth
 *
 */
public class NearestNeighbor extends ANearestNeighbor {
	
	protected double[] scaling;
	protected double[] translation;
	private List<List<Object>> trainingsData;
	
	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		//Get all unweighted Votes
		Map<Object, Double> unweightedVotes =getUnweightedVotes(subset);
		
		//Return winner class
		return getWinner(unweightedVotes);
	}
	@Override
	protected void learnModel(List<List<Object>> traindata) {
		//Save the training data
		trainingsData = traindata;	
	}
	
	@Override
	protected Map<Object, Double> getUnweightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		
		//Map for the unweighted votes results
		Map<Object, Double> result = new HashMap<Object,Double>();
		
		//Iterate over all instances in the subset
		for(Pair<List<Object>, Double> instance: subset){
			
			//Get the class attribut
			Object classAttribut = instance.getA().get(getClassAttribute());
			
			//Count the votes
			double number = 0.0;
			if(!result.containsKey(classAttribut)){
				number = 1.0;
			}
			else{
				number = result.get(classAttribut)+1.0;
				result.remove(classAttribut);
			}
			
			//Update the Result
			result.put(classAttribut, number);
		}
		return result;
	}
	@Override
	protected Map<Object, Double> getWeightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	protected Object getWinner(Map<Object, Double> votesFor) {
		//Data of the winner
		double max = 0;
		Object winnerClass = null;
		
		//Search the winner
		for(Entry entry : votesFor.entrySet()){
			if ((double)entry.getValue() >= max){
				max = (double) entry.getValue();
				winnerClass = entry.getKey();
			}
		}
		return winnerClass;
	}
	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> testdata) {
		//List to save the result
		List<Pair<List<Object>, Double>> result = new ArrayList<Pair<List<Object>, Double>>();
		
		//Calculate the distances and save it in the result list
		for(List<Object> trainingsInstance : trainingsData){
			double distance = 0;
			if(getMetric() == 0){
				distance = determineManhattanDistance(testdata, trainingsInstance);
			}
			else{
				distance = determineEuclideanDistance(testdata, trainingsInstance);
			}
			
			Pair<List<Object>, Double> resultEntry = new Pair<List<Object>, Double>(trainingsInstance, distance);
			result.add(resultEntry);
		}
		
		//Get the k nearest Instances
		Comparator<Pair<List<Object>, Double>> comp = new Comparator<Pair<List<Object>, Double>>() {

			@Override
			public int compare(Pair<List<Object>, Double> o1,
					Pair<List<Object>, Double> o2) {							
				return o1.getB().compareTo(o2.getB());
			}
		};
		
		//Sort the list and get the first k instances
		Collections.sort(result, comp);
		
		int to = Math.min(result.size(),getkNearest());
		
		//Return the list of the first k instances
		if (to > 1){
			return result.subList(0, to-1);
		}
		else{
			Pair<List<Object>, Double> resultFinal = result.get(0);
			List<Pair<List<Object>, Double>> resultList = new ArrayList<Pair<List<Object>, Double>>();
			resultList.add(resultFinal);
			return resultList;
		}

	}
	
	@Override
	protected double determineManhattanDistance(List<Object> instance1,
			List<Object> instance2) {
		//Set distance to 0
		double result = 0;
		
		//Calculate the Distance
		for(int i = 0; i<instance1.size()-1;i++){
			
			//If it is a String
			if(instance1.get(i) instanceof String){
				if(!instance1.get(i).equals(instance2.get(i))){
					result++;	
				}

			}
			else{
				//If it is numeric
				result = result + Math.abs(((double) instance1.get(i)- (double) instance2.get(i)));	
			}
		}
		return result;
	}
	@Override
	protected double determineEuclideanDistance(List<Object> instance1,
			List<Object> instance2) {
		// TODO Auto-generated method stub
		return 0;
	}
	@Override
	protected double[][] normalizationScaling() {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	protected String[] getMatrikelNumbers() {
		// TODO Auto-generated method stub
		return null;
	}

}
