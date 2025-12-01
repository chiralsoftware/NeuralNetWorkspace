package chiralsoftware.netfromscratch;

import static java.lang.Math.abs;
import static java.lang.Math.sqrt;
import static java.lang.System.out;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import jdk.incubator.vector.FloatVector;
import static jdk.incubator.vector.FloatVector.zero;
import jdk.incubator.vector.VectorSpecies;

/**
 * Represent a batch of samples and perform training and update within the batch
 */
final class BatchProcessor {
    
    private final ArrayList<Layer> layers;
    
    private TrainingTracker trainingTracker = null;
    
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final FloatVector ZERO_VEC = zero(SPECIES);

    BatchProcessor(ArrayList<Layer> layers) {
        this.layers = layers;
        weightGradientAccumulators = 
            new float[layers.size()][/* numWeightsPerNeuron */ ][ /*numNeuronsPerLayer */ ];
        biasGradientAccumulators = 
            new float[layers.size()][ /*numNeuronsPerLayer */ ];
        deltas = new float[layers.size()][];
        rawOutputs = new float[layers.size()][];
        activatedOutputs = new float[layers.size()][];
        activationDerivatives = new float[layers.size()][];
        // intitialize the accumulators
        for(int layerIndex = 0 ; layerIndex < layers.size(); layerIndex++) {
            final Layer layer = layers.get(layerIndex);
            weightGradientAccumulators[layerIndex] = new float[layer.inputSize][layer.outputSize];
            biasGradientAccumulators[layerIndex] = new float[layer.outputSize];
            deltas[layerIndex] = new float[layer.outputSize];
            
            rawOutputs[layerIndex] = new float[layer.outputSize];
            activatedOutputs[layerIndex] = new float[layer.outputSize];
            activationDerivatives[layerIndex] = new float[layer.outputSize];
        }
    }
    
    /** Zero all the accumulators */
    private void reset() {
        for(int layer = 0; layer < biasGradientAccumulators.length; layer++) {
            zeroArray(biasGradientAccumulators[layer]);
            for (float[] weightGradientAccumulator : weightGradientAccumulators[layer]) {
                zeroArray(weightGradientAccumulator);
            }
        }
        
    }
    
    static void zero2dArray(float[][] array) {
        for(int layer = 0; layer < array.length; layer++)
            zeroArray(array[layer]);
    }
    
    static void zeroArray(float[] array) {
        int i = 0;
        final int upperBound = SPECIES.loopBound(array.length);
        for (; i < upperBound; i += SPECIES.length())  ZERO_VEC.intoArray(array, i);
        // Handle remaining elements (tail)
        for (; i < array.length; i++) array[i] = 0f;
    }
    
    /** Element-wise multiple a1 and a2 and return the result in a new array
     TODO move this to a vector utilities class */
    static float[] elementwiseMultiply(float[] a1, float[] a2) {
        assert(a1.length == a2.length);
        final float[] result = new float[a1.length];
        for(int i = 0; i < a1.length; i++) result[i] = a1[i] * a2[i];
        return result;
    }
    
    // this will be an irregular array because each layer may have different
    // shape
    private final float[][][] weightGradientAccumulators;
    private final float[][] biasGradientAccumulators;
    private final float[][] deltas;
    private final float[][] rawOutputs;
    private final float[][] activatedOutputs;
    private final float[][] activationDerivatives;
    
    float process(List<Sample> samples) {
        reset();
        float totalLoss = 0;
        
        for (Sample sample : samples) {
            // Forward pass, compute raw outputs and activated outputs
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                final Layer layer = layers.get(layerIndex);

                layer.raw(layerIndex == 0 ? sample.x() : activatedOutputs[layerIndex - 1], rawOutputs[layerIndex]);
                layer.activated(rawOutputs[layerIndex], activatedOutputs[layerIndex]);
            }
            // save the error of this
            totalLoss += mseLoss(sample.target(), activatedOutputs[activatedOutputs.length - 1]);

            // Backward pass
            // layers.get(0) = input layer
            // layers.get(layers.size() - 1) = output layer
            final int outputLayerIndex = layers.size() - 1;
            {
                final Layer outputLayer = layers.get(outputLayerIndex);
                // set the delta at the output layer
                outputLayer.lossDerivative(outputLayerIndex == 0 ?
                        sample.x() : // in a single layer, gradient is vs. the input
                        activatedOutputs[outputLayerIndex - 1], // with hidden layers, gradient vs last hidden layer output
                        activatedOutputs[outputLayerIndex]
                        , sample.target(), deltas[outputLayerIndex]);
//                arraycopy(outputLayerDelta, 0, deltas[outputLayerIndex], 0, outputLayerDelta.length);
            }

            // The heart of back propagation: compute deltas for every layer
            for (int layerIndex = outputLayerIndex - 1; layerIndex >= 0; layerIndex--) {
                final Layer currentLayer = layers.get(layerIndex);
                final Layer nextLayer = layers.get(layerIndex+1);
                if(currentLayer == null) throw new NullPointerException("Layer object at layer: " + layerIndex +  " was not found");
 
                // Multiply the weighted-delta vector by the derivative of the activation function
                // evaluated at this layer’s activated output.
//                W[i][j] = weight from current neuron i to next neuron j
//                delta_next[j] = error signal for next neuron j
//                f'(z_current[i]) = derivative of the activation function at the current neuron’s pre‑activation value
        
                final float[] currentLayerWeightedDelta = new float[currentLayer.outputSize];
                if(currentLayer.outputSize != nextLayer.inputSize) 
                    throw new IllegalStateException("curent layer: " + currentLayer + " output size does not match next layer: " + 
                            nextLayer + " input size at layerIndex = " + layerIndex);
                currentLayer.activationDerivative(rawOutputs[layerIndex], activationDerivatives[layerIndex]);
                for(int j = 0 ; j < currentLayer.outputSize; j++) {
                    for (int i = 0; i < nextLayer.outputSize; i++ ) {
//                        out.println("current layer: " + currentLayer + " at index: " + layerIndex + ", i=" + i + ", j=" + j);
                        // delta_current[i] = ( sum over j of W[i][j] * delta_next[j] ) * f'(z_current[i])
                        currentLayerWeightedDelta[j] += deltas[layerIndex + 1][i] * nextLayer.weights[j][i] * activationDerivatives[layerIndex][j];
                    }
                }
                deltas[layerIndex] = currentLayerWeightedDelta;
            }
            // Convert the deltas into gradients
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                final Layer layer = layers.get(layerIndex);
                final float[] currentInput = layerIndex == 0 ? sample.x() : activatedOutputs[layerIndex - 1];
                if(currentInput.length != layer.inputSize) 
                    throw new IllegalStateException("at layer index: " + layerIndex + 
                            ", the input shape: "+ currentInput.length + " did not match the expected input shape of the layer: " + layer);
                final float[] delta = deltas[layerIndex];

                for (int j = 0; j < layer.outputSize; j++) {
                    for (int i = 0; i < layer.inputSize; i++) {
                        weightGradientAccumulators[layerIndex][i][j] += currentInput[i] * delta[j];
                    }
                }

                for (int j = 0; j < layer.outputSize; j++) {
                    biasGradientAccumulators[layerIndex][j] += delta[j];
                }
            }
        }
        
        // normalize the accumulated values
        if(trainingTracker != null) {
            zeroArray(trainingTracker.getGradientAverages());
            zeroArray(trainingTracker.getGradientStrandardDeviations());
            for(int i = 0 ;i < trainingTracker.getGradientSignBalance().length; i++)
                trainingTracker.getGradientSignBalance()[i] = 0;
        }
        for(int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            final float[][] currentLayerWeightGradientAccumulators = weightGradientAccumulators[layerIndex];
            final float[] currentLayerBiasAccumulators = biasGradientAccumulators[layerIndex];
            final Layer currentLayer = layers.get(layerIndex);
            for(int j = 0; j < currentLayer.outputSize; j++) {
                for(int i = 0; i < currentLayer.inputSize; i++) {
                    currentLayerWeightGradientAccumulators[i][j] /= samples.size();
                    if(trainingTracker != null) {
                        final float[] gradientAverages = trainingTracker.getGradientAverages();
                        gradientAverages[layerIndex] += currentLayerWeightGradientAccumulators[i][j];
                        trainingTracker.getGradientAverages()[layerIndex] += 
                            biasGradientAccumulators[layerIndex][j];
                        trainingTracker.getGradientSignBalance()[layerIndex] += 
                                currentLayerWeightGradientAccumulators[i][j] >= 0 ? 1 : -1;
                    }
                }
                currentLayerBiasAccumulators[j] /= samples.size();
            }
        }
        if(trainingTracker != null) {
            
            final float[] gradientAverages = trainingTracker.getGradientAverages();
            for(int l = 0 ; l < layers.size(); l++ ) {
                gradientAverages[l] /= 
                        layers.get(l).inputSize * layers.get(l).outputSize + layers.get(l).outputSize;
            }
            
            // do another pass to calculate the std. deviation of gradients
            // future: optimize this using Welford's single pass calculation
            for(int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                final Layer currentLayer = layers.get(layerIndex);
                final float[] gradientStandardDeviations = 
                        trainingTracker.getGradientStrandardDeviations();
                for(int i= 0; i< currentLayer.inputSize; i++) {
                    for(int j= 0; j < currentLayer.outputSize; j++) {
                        gradientStandardDeviations[layerIndex] += 
                                (weightGradientAccumulators[layerIndex][i][j] - gradientAverages[layerIndex]) *
                                (weightGradientAccumulators[layerIndex][i][j] - gradientAverages[layerIndex]) ;
                    }
                }
                for(int j = 0; j < currentLayer.outputSize; j++) {
                    gradientStandardDeviations[layerIndex] += 
                            (biasGradientAccumulators[layerIndex][j] - gradientAverages[layerIndex]) *
                            (biasGradientAccumulators[layerIndex][j] - gradientAverages[layerIndex]);
                }
                gradientStandardDeviations[layerIndex] =
                        (float) sqrt(gradientStandardDeviations[layerIndex] / 
                                (currentLayer.inputSize * currentLayer.outputSize + currentLayer.outputSize));
            }
        }
        

        // apply the accumulated gradients
        
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            final Layer layer = layers.get(layerIndex);
            final float[][] weightAccumulator = weightGradientAccumulators[layerIndex];
            final float[] biasAccumulator = biasGradientAccumulators[layerIndex];
            {
                final String accumulatorCheck = weightedGradientAccumulatorCheck(weightAccumulator);
                if(accumulatorCheck != null) {
                    out.println("Gradient failure at layer: " + layerIndex +  ": "  + accumulatorCheck);
                    for(int i = 0; i < weightAccumulator.length; i++) out.println("row:  " + i + ": " + showFloatArray(weightAccumulator[i]));
                    if(trainingTracker != null)
                        trainingTracker.setMessage("Gradient failure at layer: " + layerIndex);
                }
            }

            for (int i = 0; i < layer.inputSize; i++) {
                for (int j = 0; j < layer.outputSize; j++) {
                    layer.weights[i][j] -= Network.learningRate * weightAccumulator[i][j];
                }
            }

            for (int j = 0; j < layer.outputSize; j++) {
                layer.biases[j] -= Network.learningRate * biasAccumulator[j];
            }
        }
        return totalLoss / samples.size();
    }
    
    private static float mseLoss(float[] output, float[] target) {
        if(output.length != target.length) throw new IllegalArgumentException("output length and target length must be equal");
        float result = 0;
        for (int i = 0; i < output.length ; i++) result += (output[i] - target[i]) * (output[i] - target[i]);
        return result / output.length;
    }

    private static boolean almostEqualFloats(float f1, float f2) {
        return abs(f1 - f2) < 1e-9;
    }
    
    private static boolean almostEqualFloatArrays(float[] f1, float[] f2) {
        if(f1.length != f2.length) throw new IllegalArgumentException("f1 length: " + f1.length + " did not equal f2 length: " + f2.length);
        for(int i = 0; i < f1.length; i++) 
            if(! almostEqualFloats(f1[i], f2[i])) return false;
        return true;
    }
    
    private static boolean almostZero(float[] f1) {
        for(int i = 0;i < f1.length; i++)
            if(abs(f1[i]) > 1e-10 ) return false;
        return true;
    }
    
    private static boolean overThreshold(float[] f1, float threshold) {
        for(int i = 0; i < f1.length; i++) if(abs(f1[i]) > threshold) return true;
        return false;
    }
    
    /** Perform some sanity checks to see if something is wrong in the learning process */
    private static String weightedGradientAccumulatorCheck(float[][] weightedGradientAccumulators) {
        boolean fail = true;
        for(int i = 0; i < weightedGradientAccumulators.length; i++) 
            if(! almostZero(weightedGradientAccumulators[i])) fail = false; 
        if(fail) return "Gradient arrays were almost all zero";
        for(int i = 0 ; i < weightedGradientAccumulators.length; i++) {
            if(overThreshold(weightedGradientAccumulators[i], 1e6f)) 
                return "row " + i + " had a value over the threshold";
        }
        for(int i = 1; i< weightedGradientAccumulators.length; i++) 
            if(! almostEqualFloatArrays(weightedGradientAccumulators[0], weightedGradientAccumulators[i])) return null;
        return "all rows are nearly equal";
    }
    
    private static String showFloatArray(float[] f1) {
        final DecimalFormat df = new DecimalFormat("0.000000");
        final StringBuilder sb = new StringBuilder("[");
        for(int i = 0; i < f1.length; i++) {
            sb.append(df.format(f1[i]));
            if(i != f1.length - 1) sb.append(", ");
        }
        return sb.append("]").toString();
    }
    
    public void setTrainingTracker(TrainingTracker trainingTracker) {
        this.trainingTracker = trainingTracker;
    }
    
}
