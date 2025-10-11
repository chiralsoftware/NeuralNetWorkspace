package chiralsoftware.netfromscratch;

/**
 * An interface for all activation functions
 */
public interface ActivationFunction {
    
    public float[] activate(float[] input);
    
    public float derivative(float output);
    
}
