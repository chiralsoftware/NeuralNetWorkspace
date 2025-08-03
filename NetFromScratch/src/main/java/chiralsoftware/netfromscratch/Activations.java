package chiralsoftware.netfromscratch;

import static java.lang.Math.exp;

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
    

}
