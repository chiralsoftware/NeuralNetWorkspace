package chiralsoftware.netfromscratch;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.max;

/** 
 f(z_i) = exp(z_i) / sum(exp(z_j)) for all j in the layer
*/
final class SoftMaxLayer extends Layer {
    
    SoftMaxLayer(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    public float[] computeGradient(float[] prediction, float[] target) {
        float[] gradient = new float[prediction.length];
        for (int i = 0; i < prediction.length; i++) {
            gradient[i] = prediction[i] - target[i];
        }
        return gradient;
    }

    @Override
    void update(float[] accumulatedWeightedGradient, float[]  accumulatedBiasGradient) {
    }

    @Override
    protected float[] activated(float[] raw) {
        if(raw.length != outputSize ) 
            throw new IllegalArgumentException("raw[] length was: " + raw.length + 
                    " but expected outputSize: " + outputSize);
        float max = Float.NEGATIVE_INFINITY;
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

    @Override
    public float loss(float[] input, float[] target) {
        // very inefficient implementation - we need to find the entire activation again
        final float[] prediction = activated(raw(input));
        int targetIndex = -1;
        for(int i = 0; i < target.length; i++) {
            if(target[i] > 0.9999f) {
                targetIndex = i; break;
            }
        }
        if(targetIndex == -1) {
            throw new IllegalStateException("attempting to use a SoftMax loss function "
                    + "on a target that doesn't have a value of 1");
        }
            
        final float p = max(prediction[targetIndex], 1e-7f); // this is to avoid log(0)
        return (float) -log(p);
   }

    /** very inefficient calculation of the gradient */
    protected float[] gradient(float[] input, float[] target) {
        if(input.length !=  inputSize)
            throw new IllegalArgumentException("the length of the input array: " + 
                    input.length + " did not equal the number of weights: " + inputSize);
        if(target.length != outputSize) 
            throw new IllegalArgumentException("the length of the target array: " + target.length + 
                    " did not equal the length of the output size (number of neurons): " + outputSize);
        final float[] prediction = activated(raw(input));
        final float[] grad = new float[prediction.length];
        for (int i = 0; i < prediction.length; i++) {
            grad[i] = prediction[i] - target[i];
        }
        return grad;        
    }
}
