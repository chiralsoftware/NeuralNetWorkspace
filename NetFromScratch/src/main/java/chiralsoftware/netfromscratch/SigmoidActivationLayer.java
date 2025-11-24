package chiralsoftware.netfromscratch;

import static java.lang.Math.exp;
import static java.lang.Math.sqrt;
import java.util.Random;

/**
 * Activation layer using Sigmoid
 */
final class SigmoidActivationLayer extends ActivationLayer {

    SigmoidActivationLayer(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }
    
    /** 
     
   Xavier / Glorot Initialization (for sigmoid or tanh)

    Goal: keep activations in a healthy range, avoid vanishing/exploding gradients.

    Formula: sample weights uniformly from [-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out))] 
    * where fan_in = number of inputs to the layer, fan_out = number of outputs.

    Example: for your 768 inputs and 32 outputs, range is about [-0.085, 0.085].
    */
    @Override
    void initRandom() {
        final Random random = new Random();
        final float bound = (float) sqrt((6f / (inputSize + outputSize)));
      for(int i = 0; i < inputSize; i++) {
          for(int j = 0; j < outputSize; j++) {
              weights[i][j] = random.nextFloat() * bound * 2 - bound;
          }
      }  
    }

    @Override
    protected float activation(float x) {
        return sigmoid(x);
    }

    @Override
    protected float activationDerivative(float x) {
        return sigmoid(x) * (1f - sigmoid(x));
    }

    private float sigmoid(float x) {
        if (x < -40) return 0; // Avoid underflow
        if (x > 40) return 1;  // Avoid overflow
        return 1f / (1f + (float)exp(-x));        
    }
    
    @Override
    protected float[] lossDerivative(float[] input, float[] target, float[] result) {
        if (input.length != inputSize)
            throw new IllegalArgumentException("input length: " + 
                    input.length + " does not equal layer input size: " + inputSize);
        if(target.length != outputSize) 
            throw new IllegalArgumentException("target.length: " + target.length + 
                    " does not match outputSize: " + outputSize);
        if(result.length != outputSize) 
            throw new IllegalArgumentException("result.length " + result.length + 
                    " does not match the output size: " + outputSize);
        
        raw(input, result);
        final float[] activatedOutput = activated(result);
        final float[] gradient = new float[outputSize];

        for (int i = 0; i < outputSize; i++) {
            final float error = activatedOutput[i] - target[i];
            final float sigmoidDerivative = activatedOutput[i] * (1 - activatedOutput[i]);
            gradient[i] = error * sigmoidDerivative;
        }

        return gradient;
    }
    
}
