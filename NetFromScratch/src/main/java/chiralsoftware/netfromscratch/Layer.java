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
        weights = new float[outputSize][inputSize];
        biases = new float[outputSize];
        this.outputSize = outputSize;
        this.inputSize = inputSize;
    }
    
    /** Set random initialization values */
    void initRandom() {
        final Random random = new Random();
        for(int i = 0; i < outputSize; i++) {
            for(int j = 0; j < inputSize; j++) {
                weights[i][j] = random.nextFloat() * 0.1f;
            }
            biases[i] = 1f - random.nextFloat(0.5f);
        }
    }
    
    final float[] raw(float[] input) {
        if(input.length != inputSize) 
            throw new IllegalArgumentException("the input length: " + input.length +
                    " was not equal to the number of weights: " + inputSize);
//        i = output neuron (row). i always ranges from 0 to biases.length
//        j = input feature (column) j always ranges from 0 to weights[0].length
        final float[] result = new float[biases.length];
        for(int i = 0; i < outputSize; i++) {
            for(int j = 0; j < inputSize; j++) {
                result[i] += input[j] * weights[i][j];
            }
            result[i] += biases[i];
        }
        return result;
    }

    protected abstract float[] gradient(float[] input, float[] target);
    
    protected abstract float[] activated(float[] raw);
    
    protected abstract float loss(float[] input, float[] target) ;
    
}
