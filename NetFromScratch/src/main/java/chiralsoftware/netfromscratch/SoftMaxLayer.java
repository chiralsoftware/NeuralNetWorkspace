package chiralsoftware.netfromscratch;

import static java.lang.Float.NEGATIVE_INFINITY;
import static java.lang.Math.exp;

/** 
 f(z_i) = exp(z_i) / sum(exp(z_j)) for all j in the layer
*/
final class SoftMaxLayer extends Layer {
    
    SoftMaxLayer(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    @Override
    float[] activationDerivative(float[] input) {
        throw new 
        UnsupportedOperationException("SoftMax doesn't support activationDerivative and shouldn't be a hidden layer");
    }

    @Override
    protected float[] activated(float[] raw) {
        if(raw.length != outputSize ) 
            throw new IllegalArgumentException("raw[] length was: " + raw.length + 
                    " but expected outputSize: " + outputSize);
        float max = NEGATIVE_INFINITY;
        for (float val : raw) {
            if (val > max) max = val;
        }   

        final float[] numerator = new float[outputSize];
        float denominator = 0f;
        for (int i = 0; i < outputSize; i++) {
            // Subtract max for numerical stability (avoids large exponentials)
            numerator[i] = (float) exp(raw[i] - max);
            denominator += numerator[i];
        }

        final float[] result = new float[outputSize];
        for (int i = 0; i < outputSize; i++) 
            result[i] = numerator[i] / denominator;

        return result;
    }

    /** very inefficient calculation of the gradient */
    @Override
    protected float[] lossDerivative(float[] input, float[] target, float[] result) {
        if(input.length !=  inputSize)
            throw new IllegalArgumentException("the length of the input array: " + 
                    input.length + " did not equal the number of weights: " + inputSize);
        if(target.length != outputSize) 
            throw new IllegalArgumentException("the length of the target array: " + target.length + 
                    " did not equal the length of the output size (number of neurons): " + outputSize);
        if(result.length != outputSize) 
            throw new IllegalArgumentException("result length: " + result.length + 
                    " does not match output size: " + outputSize);
        raw(input, result);
        final float[] prediction = activated(result);
        final float[] grad = new float[prediction.length];
        for (int i = 0; i < prediction.length; i++) {
            grad[i] = prediction[i] - target[i];
        }
        return grad;        
    }
}
