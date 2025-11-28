package chiralsoftware.netfromscratch;

import static java.lang.Float.NEGATIVE_INFINITY;
import static java.lang.Math.exp;

/** 
 f(z_i) = exp(z_i) / sum(exp(z_j)) for all j in the layer
*/
public final class SoftMaxLayer extends Layer {
    
    public SoftMaxLayer(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    @Override
    void activationDerivative(float[] input, float[] result) {
        throw new 
        UnsupportedOperationException("SoftMax doesn't support activationDerivative and shouldn't be a hidden layer");
    }

    @Override
    protected void activated(float[] raw, float[] result) {
        if(raw.length != outputSize ) 
            throw new IllegalArgumentException("raw[] length was: " + raw.length + 
                    " but expected outputSize: " + outputSize);
        if(result.length != outputSize) 
            throw new IllegalArgumentException("the result length: " + 
                    result.length + " was not equal to the output size: "+ outputSize);

        float max = NEGATIVE_INFINITY; for (float val : raw) if (val > max) max = val;

        float denominator = 0f;
        for (int i = 0; i < outputSize; i++) {
            // Subtract max for numerical stability (avoids large exponentials)
            result[i] = (float) exp(raw[i] - max);
            denominator += result[i];
        }

        for (int i = 0; i < outputSize; i++) result[i] /= denominator;
    }

    /** very inefficient calculation of the gradient */
    @Override
    protected void lossDerivative(float[] input, float[] activated, float[] target, float[] result) {
        if(input.length !=  inputSize)
            throw new IllegalArgumentException("the length of the input array: " + 
                    input.length + " did not equal the number of weights: " + inputSize);
        if(target.length != outputSize) 
            throw new IllegalArgumentException("the length of the target array: " + target.length + 
                    " did not equal the length of the output size (number of neurons): " + outputSize);
        if(result.length != outputSize) 
            throw new IllegalArgumentException("result length: " + result.length + 
                    " does not match output size: " + outputSize);

//        raw(input, result);
//        final float[] prediction = new float[result.length];
//        activated(result, prediction);
//        for (int i = 0; i < prediction.length; i++) {
//            result[i] = prediction[i] - target[i];
//        }
        
        for (int i = 0; i < activated.length; i++) {
            result[i] = activated[i] - target[i];
        }
    }
}
