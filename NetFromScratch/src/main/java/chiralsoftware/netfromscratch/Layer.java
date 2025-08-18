package chiralsoftware.netfromscratch;

import static java.lang.System.out;

/**
 * A layer of neurons
 */
public record  Layer(Neuron[] neurons, ActivationFunction activation) {
    
    /** calculate a[] , which is activation(z) */
    float[] forward(float[] x) {
        final float[] result = rawOutputs(x);
        for(int i = 0; i < result.length; i++) 
            result[i] = activation.activate(result[i]);
        return result;        
    } 
    
    /** calculate z[] which is the raw result of the neurons in the layer */
    float[] rawOutputs(float[] x) {
        final float[] result = new float[neurons.length];
        for(int i = 0; i < neurons.length; i++ ) {
            result[i] = activation.activate(neurons[i].calculateRaw(x));
        }
        return result;        
    }
    
    void update(float[] lossGradients, float[] x) {
        for(int i = 0; i < neurons.length;i++) {
            neurons[i].adjust(lossGradients[i], x);
        }
    }

}
