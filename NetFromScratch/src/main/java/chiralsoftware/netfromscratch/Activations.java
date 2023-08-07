package chiralsoftware.netfromscratch;

import static java.lang.Math.exp;
import static java.lang.Math.max;

/**
 * Various activation functions to use
 */
final class Activations {
    
    private Activations() { throw new RuntimeException(); }
    
    public static final Activation sigmoidLogistic = new Activation() {
        @Override
        public float activation(float activation) {
            return (float) (1 / (1 + exp(-1 * activation)));
        }

        @Override
        public float derivative(float output) {
            // see: https://en.wikipedia.org/wiki/Logistic_function#Derivative
            // the derivative is very simple based on the calculated output
            return output * (1 - output);
        }
    };
    
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
