package chiralsoftware.netfromscratch;

/**
 * An interface for all activation functions
 */
public interface Activation {
    
    public float activation(float input);
    
    public float derivative(float output);
    
}
