package chiralsoftware.netfromscratch;

import java.util.Arrays;
import java.util.Random;

/**
 * Represent a single neuron, which includes a set of weights
 * and a bias. 
 */
public class Neuron {
    
    private final float[] weights;
    private float bias;
    private float z_cache = 0;
    
    static Neuron random(int size) {
        final Random random = new Random();
        final Neuron n = new Neuron(size);
        for(int i = 0; i < n.weights.length; i++) {
            n.weights[i] = random.nextFloat() * 0.1f;
        }
        n.bias = 1f - random.nextFloat(0.5f);
        return n;
    }
    
    private Neuron(int size) {
        this.weights = new float[size];
    }
            
    /** Apply the weights and bias to the input. Return result is raw - no activation function is applied. */
    float calculateRaw(float[] input) {
        if(input == null) throw new NullPointerException("can't process a null input");
        if(input.length != weights.length) 
            throw new IllegalArgumentException("input array length: " + 
                    input.length + " did not match weights array length: " + weights.length);
        
        float result = 0;
        for(int i = 0; i < weights.length; i++)
            result += weights[i] * input[i];
        z_cache = result + bias;
        return z_cache;
    }
    
    @Override
    public String toString() {
        return "bias: " + bias + ", weights: " + Arrays.toString(weights);
    }

    // 
    void adjust(float f, float[] input) {
        if(weights.length != input.length)
            throw new IllegalArgumentException("input length: " + input.length + 
                    " and weights[] length: " + weights.length);
        
        for(int i = 0; i < weights.length; i++ ) {
            weights[i] -= f * input[i] * 0.01; 
        }
        // dL_i/db = loss'(activation(z)) * activation'(w_i * x_i + b) 
        bias -= f * 0.01f;
    }
    
}
