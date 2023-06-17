package chiralsoftware.netfromscratch;

import static java.lang.Math.exp;
import static java.lang.Math.max;

/**
 * Various activation functions to use
 */
final class Activations {
    
    private Activations() { throw new RuntimeException(); }
    
    /** 1 / (1 + exp(-x)) */
    static float sigmoidLogistic(float x) {
        return (float) (1 / (1 + exp(-1 * x)));
    }
    
    static float swish(float x) {
        return (float) (x / (1 + exp(-1 * x)));
    }    
    
    static float tanh(float x) {
        return (float) ((exp(x) - exp(-1 * x)) / (exp(x) + exp(-1 * x)));
    }
    
    static float relu(float x) {
        return max(0,x);
    }
}
