package chiralsoftware.netfromscratch;

import static java.lang.Math.exp;

/**
 * Various activation functions to use
 */
final class Activations {
    
    private Activations() { throw new RuntimeException(); }
    
    public static final ActivationFunction sigmoidLogistic = new ActivationFunction() {
        @Override
        public float[] activate(float[] activation) {
            final float[] result = new float[activation.length];
            for(int i = 0; i < activation.length; i++) {
                result[i] = (float) (1 / (1 + exp(-1 * activation[i])));
            }
            return result;
        }

        @Override
        public float derivative(float output) {
            // see: https://en.wikipedia.org/wiki/Logistic_function#Derivative
            // the derivative is very simple based on the calculated output
            return output * (1 - output);
        }
    };
    

}
