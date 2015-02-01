package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
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
	
	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	protected void learnModel(List<List<Object>> traindata) {
		// TODO Auto-generated method stub
		
	}
	@Override
	protected Map<Object, Double> getUnweightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	protected Map<Object, Double> getWeightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	protected Object getWinner(Map<Object, Double> votesFor) {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> testdata) {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	protected double determineManhattanDistance(List<Object> instance1,
			List<Object> instance2) {
		// TODO Auto-generated method stub
		return 0;
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
