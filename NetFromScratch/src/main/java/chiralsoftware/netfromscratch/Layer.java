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
    
    abstract void update(float[] accumulatedWeightedGradient, float[]  accumulatedBiasGradient);
    
    Layer(int inputSize, int outputSize) {
        weights = new float[inputSize][outputSize];
        biases = new float[outputSize];
        this.outputSize = outputSize;
        this.inputSize = inputSize;
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
    
    final float[] raw(float[] input) {
        if (input.length != inputSize)
            throw new IllegalArgumentException("Input length " + input.length +
                " does not match expected input size " + inputSize);

        final float[] result = new float[outputSize];
        for (int j = 0; j < outputSize; j++) {
            for (int i = 0; i < inputSize; i++) {
                result[j] += input[i] * weights[i][j];
            }
            result[j] += biases[j];
        }
        return result;
    }

    protected abstract float[] gradient(float[] input, float[] target);
    
    protected abstract float[] activated(float[] raw);
    
    protected abstract float loss(float[] input, float[] target) ;
    
}
