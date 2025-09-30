package chiralsoftware.netfromscratch;

/**
 * A layer of neurons
 */
abstract class  Layer {
    
    protected final Neuron[] neurons;
    abstract float[] forward(float[] input);
    abstract float computeLoss(float[] prediction, float[] target);
    abstract float[] computeGradient(float[] prediction, float[] target);
    abstract void update(float[] accumulatedWeightedGradient, float[]  accumulatedBiasGradient);
    
    Layer(Neuron[] neurons) {
        this.neurons = neurons;
    }
    
    protected float[] lastRaw = null;
    protected float[] lastInput = null;
    
    float[] raw(float[] input) {
        final float[] result = new float[neurons.length];
        for(int i = 0; i < neurons.length; i++) {
            result[i] = neurons[i].calculateRaw(input);
        }
        lastRaw = result;
        lastInput = input;
        return result;
    }
    
}
