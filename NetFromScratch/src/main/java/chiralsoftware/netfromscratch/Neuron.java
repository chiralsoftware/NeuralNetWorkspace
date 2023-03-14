package chiralsoftware.netfromscratch;

import java.util.Arrays;
import java.util.Random;

/**
 * Represent a single neuron, which includes a set of weights
 * and a bias
 */
public final class Neuron {
    
    private final float[] weights;
    private float bias;
    
    Neuron(int weights) {
        this.weights = new float[weights];
    }
    
    /** Apply the weights to byte input. This is for the MNIST input layer, which are bytes */
    float calculate(byte[] input) {
        if(input == null) throw new NullPointerException("can't process a null input");
        if(input.length != weights.length) 
            throw new IllegalArgumentException("input array length: " + 
                    input.length + " did not match weights array length: " + weights.length);
        
        float result = 0;
        for(int i = 0; i < weights.length; i++)
            result += weights[i] * input[i];
        return result + bias;
    }
    
    /** Apply the weights and bias and sigmoid to calculate the output */
    float calculate(float[] input) {
        if(input == null) throw new NullPointerException("can't process a null input");
        if(input.length != weights.length) 
            throw new IllegalArgumentException("input array length: " + 
                    input.length + " did not match weights array length: " + weights.length);
        
        float result = 0;
        for(int i = 0; i < weights.length; i++)
            result += weights[i] * input[i];
        return result + bias;
    }
    
    void initialize() {
        final Random random = new Random();
        for(int i = 0; i < weights.length; i++) {
            weights[i] = random.nextFloat(1)* 0.1f;
        }
        bias = random.nextFloat(1)* 0.1f;;
    }
    
    @Override
    public String toString() {
        return "bias: " + bias + ", weights: " + Arrays.toString(weights);
    }
    
    
}
