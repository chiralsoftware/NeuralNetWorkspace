package chiralsoftware.netfromscratch;

import static java.lang.Math.exp;

/**
 * Activation layer using Sigmoid
 */
final class SigmoidActivationLayer extends ActivationLayer {

    SigmoidActivationLayer(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    @Override
    protected float[] gradient(float[] input, float[] target) {
        if (input.length != inputSize)
            throw new IllegalArgumentException("input length: " + 
                    input.length + " does not equal layer input size: " + inputSize);
        if(target.length != outputSize) 
            throw new IllegalArgumentException("target.length: " + target.length + 
                    " does not match outputSize: " + outputSize);

        final float[] activatedOutput = activated(raw(input));
        final float[] gradient = new float[outputSize];

        for (int i = 0; i < outputSize; i++) {
            final float error = activatedOutput[i] - target[i];
            final float sigmoidDerivative = activatedOutput[i] * (1 - activatedOutput[i]);
            gradient[i] = error * sigmoidDerivative;
        }

        return gradient;
    }

    @Override
    protected float[] activated(float[] raw) {
        if(raw.length != outputSize) 
            throw new IllegalArgumentException("raw length: " + raw.length + 
                    " does not equal the output size: " + outputSize);
        final float[] result = new float[outputSize];
        for(int i = 0;i < raw.length; i++) {
            result[i] = sigmoid(raw[i]);
        }
        return result;
    }
    
    private float sigmoid(float x) {
        if (x < -40) return 0; // Avoid underflow
        if (x > 40) return 1;  // Avoid overflow
        return 1f / (1f + (float)exp(-x));        
    }

    @Override
    protected float loss(float[] input, float[] target) {
        if (input.length != inputSize || target.length != outputSize)
            throw new IllegalArgumentException("Input or target size mismatch");

        final float[] activatedOutput = activated(raw(input));
        float sum = 0;

        for (int i = 0; i < outputSize; i++) {
            final float error = activatedOutput[i] - target[i];
            sum += error * error;
        }

        return sum / outputSize;
    }
    
}
