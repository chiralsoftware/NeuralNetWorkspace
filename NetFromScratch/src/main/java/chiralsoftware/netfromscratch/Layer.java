package chiralsoftware.netfromscratch;

/**
 * A layer of neurons
 */
abstract class  Layer {
    
    protected Neuron[] neurons;
    abstract float[] forward(float[] input);
    abstract float computeLoss(float[] prediction, float[] target);
    abstract float[] computeGradient(float[] prediction, float[] target);
    abstract void update(float[] gradient, float[] input);
    
}
