package chiralsoftware.netfromscratch;

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
    final void activated(float[] raw, float[] result) {
        if(raw.length != outputSize) 
            throw new RuntimeException("raw.length was: " + raw.length + 
                    ", should be inputSize = " + outputSize);
        if(result.length != outputSize)
            throw new RuntimeException("result length: " + 
                    result.length + " is not equal to output size: " + outputSize);
        
        for(int j = 0; j < outputSize; j++) result[j] = activation(raw[j]);
    }
    
    /** Take the raw output z and return derivative of the activation function */
    @Override
    final void activationDerivative(float[] raw, float[] result) {
        if(raw.length != outputSize)
            throw new IllegalArgumentException("the raw.length was: " + raw.length + 
                    " was not equal to outputSize: " + outputSize);
        if(result.length != outputSize)
            throw new IllegalArgumentException("result length: " + 
                    result.length + " was not equal to outputSize: " + outputSize);
        for(int i = 0; i < outputSize; i++) 
            result[i] = activationDerivative(raw[i]);
    }
    
}
