package chiralsoftware.netfromscratch;

import static java.lang.System.Logger.Level.INFO;

/**
 */
abstract class ActivationLayer extends Layer {

    private static final System.Logger LOG = System.getLogger(ActivationLayer.class.getName());

    ActivationLayer(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }
    abstract protected float activationDerivative(float x);
    abstract protected float activation(float x);
    
    /** Take the raw output z and apply the activation function to it */
    @Override
    final float[] activated(float[] raw) {
        if(raw.length != outputSize) 
            throw new RuntimeException("raw.length was: " + raw.length + 
                    ", should be inputSize = " + outputSize);
        
        final float[] result = new float[outputSize];
        for(int j = 0; j < outputSize; j++) result[j] = activation(raw[j]);
        return result;
    }
    
    /** Take the raw output z and return derivative of the activation function */
    @Override
    final float[] activationDerivative(float[] raw) {
        if(raw.length != outputSize)
            throw new IllegalArgumentException("the raw.length was: " + raw.length + 
                    " was not equal to outputSize: " + outputSize);
        final float[] result = new float[outputSize];
        for(int i = 0; i < outputSize; i++) 
            result[i] = activationDerivative(raw[i]);
        return result;
    }
    
}
