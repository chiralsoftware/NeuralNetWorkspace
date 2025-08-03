package chiralsoftware.netfromscratch;

/**
 * A layer of neurons
 */
public record  Layer(Neuron[] neurons, Activation activation) {
    
    float[] calculate(float[] input) {
        final float[] result = new float[neurons.length];
        for(int i = 0; i < neurons.length; i++ ) {
            result[i] = activation.activation(neurons[i].calculate(input));
        }
        return result;
    }
    
}
