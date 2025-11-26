package chiralsoftware.netfromscratch;

import java.util.Random;

/**
 * A layer of neurons. There is no Neuron class anymore. The neurons
 * are held directly in the layer.
 */
abstract class  Layer {
    
    protected final float[][] weights; // [outputSize][inputSize]
    protected final float[] biases; // [outputSize]
    
    public final int outputSize;
    public final int inputSize;
    
    Layer(int inputSize, int outputSize) {
        weights = new float[inputSize][outputSize];
        biases = new float[outputSize];
        this.outputSize = outputSize;
        this.inputSize = inputSize;
    }
    
    @Override
    public String toString() {
        return getClass().getSimpleName() + " inputSize=" + inputSize + ", outputSize=" + outputSize;
    }
    /** Set random initialization values */
    void initRandom() {
        final Random random = new Random();
        for(int i = 0; i < inputSize; i++) {
            for(int j = 0; j < outputSize; j++) {
                weights[i][j] = random.nextFloat() * 0.1f;
            }
        }
        for(int j = 0; j < outputSize; j++)
            biases[j] = 1f - random.nextFloat(0.5f);

    }
    
    /** Calculate z, the raw output of this layer. 
     z = dot_product(weights, inputs) + bias
     */
    final void raw(float[] input, float[] result) {
        if (input.length != inputSize)
            throw new IllegalArgumentException("Input length " + input.length +
                " does not match expected input size " + inputSize);
        if(result.length != outputSize) 
            throw new IllegalArgumentException("result array size: " + result.length + " does not match output size: " + outputSize);

        for (int j = 0; j < outputSize; j++) {
            result[j] = 0;
            for (int i = 0; i < inputSize; i++) result[j] += input[i] * weights[i][j];
            result[j] += biases[j];
        }
    }

    void lossDerivative(float[] input, float[] activated, float[] target, float[] result) {
        throw new UnsupportedOperationException("this class doesn't support lossDerivative");
    }
    
    void activationDerivative(float[] raw, float[] result) {
        throw new UnsupportedOperationException("this class doesn't support activationDerivative");
    }
    
    /** Given the results of raw, return it with the activation applied. 
     @param raw length == outputSize */
    abstract void activated(float[] raw, float [] result);
    
    
}
