package chiralsoftware.netfromscratch;

import java.util.Arrays;
import java.util.Random;

/**
 * Represent a single neuron, which includes a set of weights
 * and a bias. 
 */
public record Neuron(float[] weights,  float[] bias) {
            
    /** Apply the weights and bias to the input. Return result is raw - no activation function is applied. */
    float calculate(float[] input) {
        if(input == null) throw new NullPointerException("can't process a null input");
        if(input.length != weights.length) 
            throw new IllegalArgumentException("input array length: " + 
                    input.length + " did not match weights array length: " + weights.length);
        
        float result = 0;
        for(int i = 0; i < weights.length; i++)
            result += weights[i] * input[i];
        return result + bias[0];
    }
    
    static Neuron random(int size) {
        final Random random = new Random();
        final Neuron n = new Neuron(new float[size], new float[] { 0 });
        for(int i = 0; i < n.weights.length; i++) {
            n.weights[i] = random.nextFloat() * 0.1f;
        }
        n.bias[0] = 0.1f - random.nextFloat(0.2f);
        return n;
    }
    
    @Override
    public String toString() {
        return "bias: " + bias[0] + ", weights: " + Arrays.toString(weights);
    }
    
}
