package chiralsoftware.netfromscratch;

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
    
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final FloatVector ZERO_VEC = zero(SPECIES);

    BatchProcessor(ArrayList<Layer> layers) {
        this.layers = layers;
        weightGradientAccumulators = 
            new float[layers.size()][/* numWeightsPerNeuron */ ][ /*numNeuronsPerLayer */ ];
        biasGradientAccumulators = 
            new float[layers.size()][ /*numNeuronsPerLayer */ ];
        // intitialize the accumulators
        for(int layerIndex = 0 ; layerIndex < layers.size(); layerIndex++) {
            final Layer layer = layers.get(layerIndex);
            weightGradientAccumulators[layerIndex] = new float[layer.inputSize][layer.outputSize];
            biasGradientAccumulators[layerIndex] = new float[layer.outputSize];
        }
    }
    
    /** Zero all the accumulators */
    private void reset() {
        for(int layer = 0; layer < biasGradientAccumulators.length; layer++) {
            zeroArray(biasGradientAccumulators[layer]);
            for(int i = 0; i < weightGradientAccumulators[layer].length; i++) {
                zeroArray(weightGradientAccumulators[layer][i]);
            }
        }
        
    }
    
    static void zeroArray(float[] array) {
        int i = 0;
        final int upperBound = SPECIES.loopBound(array.length);
        for (; i < upperBound; i += SPECIES.length())  ZERO_VEC.intoArray(array, i);
        // Handle remaining elements (tail)
        for (; i < array.length; i++) array[i] = 0f;
    }
    
    // this will be an irregular array because each layer may have different
    // shape
    private final float[][][] weightGradientAccumulators;
    private final float[][] biasGradientAccumulators;
    
    float process(List<Sample> samples) {
        reset();
        float totalLoss = 0;
        
        for (Sample sample : samples) {
            final float[] input = sample.x();
            final float[] target = sample.target();

            final float[][] rawOutputs = new float[layers.size()][];
            final float[][] activatedOutputs = new float[layers.size()][];
            float[] currentInput = input;

            // Forward pass
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                final Layer layer = layers.get(layerIndex);

                final float[] raw = layer.raw(currentInput);
                final float[] activated = layer.activated(raw);

                rawOutputs[layerIndex] = raw ;
                activatedOutputs[layerIndex] = activated;

                currentInput = activated;
            }
            // save the error of this
            totalLoss += mseLoss(sample.target(), activatedOutputs[activatedOutputs.length - 1]);

            // Backward pass
            // layers.get(0) = input layer
            // layers.get(layers.size() - 1) = output layer

            float[] nextDelta = layers.getLast().gradient(layers.size() == 1 ?
                    sample.x() : // in a single layer, gradient is vs. the input
                    activatedOutputs[activatedOutputs.length - 2] // with hidden layers, gradient vs last hidden layer output
                    , target);

            final int outputLayerIndex = layers.size() - 1;
            final Layer outputLayer = layers.get(outputLayerIndex);
            {
                final float[] prevActivation = outputLayerIndex == 0 ? sample.x() : activatedOutputs[outputLayerIndex - 1];

                final float[][] outputLayerGradients = weightGradientAccumulators[outputLayerIndex];
                final float[] outputBiasGradients = biasGradientAccumulators[outputLayerIndex];

                for (int i = 0; i < outputLayer.outputSize; i++) {
                    for (int j = 0; j < outputLayer.inputSize; j++) {
                        outputLayerGradients[j][i] += prevActivation[j] * nextDelta[i];
                    }
                    outputBiasGradients[i] += nextDelta[i];
                }
            }

            for (int layerIndex = layers.size() - 2; layerIndex >= 0; layerIndex--) {
                final Layer currentLayer = layers.get(layerIndex);
                final Layer nextLayer = layers.get(layerIndex + 1);
                
                final float[] delta = new float[currentLayer.outputSize];
                for(int j = 0; j < currentLayer.outputSize; j++) {
                    for(int i = 0; i < nextDelta.length; i++) {
                        delta[j] += nextLayer.weights[j][i] * nextDelta[i];
                    }
                    delta[j] *= ((ActivationLayer) currentLayer).activationDerivative(activatedOutputs[layerIndex][j]);
                }
                final float[][] currentLayerGradients = weightGradientAccumulators[layerIndex];
                if(currentLayerGradients.length != currentLayer.inputSize) 
                    throw new IllegalStateException("at layer index: " + layerIndex + 
                            " the weightedGradientAccumulator length: " + currentLayerGradients.length + 
                            " doesn't match the layer input length: " + currentLayer.inputSize);

                final float[] prevActivation = (layerIndex == 0) ? sample.x() : activatedOutputs[layerIndex - 1];

                for(int i = 0; i < currentLayer.outputSize; i++) {
                    for(int j = 0; j < currentLayer.inputSize; j++) {
                        currentLayerGradients[j][i] += 
                                prevActivation[j] * delta[i];
                    }
                    biasGradientAccumulators[layerIndex][i] += delta[i];
                }
            }
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                final Layer layer = layers.get(layerIndex);
                final float[][] weightGradients = weightGradientAccumulators[layerIndex];
                if(weightGradients == null) throw new NullPointerException("weightGradients was null at layer index: " + layerIndex);
                final float[] biasGradients = biasGradientAccumulators[layerIndex];

                for (int j = 0; j < layer.outputSize; j++) {
                    for (int i = 0; i < layer.inputSize; i++) {
//        weights = new float[inputSize][outputSize];
//        biases = new float[outputSize];
                        layer.weights[i][j] -= Network.learningRate * weightGradients[i][j] / samples.size();
                    }
                    layer.biases[j] -= Network.learningRate * biasGradients[j] / samples.size();
                }
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

    /** not used anymore */
    private float processOneLayer(List<Sample> samples) {
        final Layer layer = layers.get(0); // only one layer for now
        final float[][] weightGradSum = new float[layer.outputSize][layer.inputSize];
        final float[] biasGradSum = new float[layer.outputSize];
        float loss = 0;
        for(int sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++) {
            final Sample sample = samples.get(sampleIndex);
            final float[] gradient = layer.gradient(sample.x(), sample.target());
            loss += layer.loss(sample.x(), sample.target());
            for (int i = 0; i < layer.outputSize; i++) {
                for (int j = 0; j < layer.inputSize; j++) {
                    weightGradSum[i][j] += gradient[i] * sample.x()[j];
                }
                biasGradSum[i] += gradient[i];
            }
        }

        for (int i = 0; i < layer.outputSize; i++) {
            for (int j = 0; j < layer.inputSize; j++) {
                layer.weights[i][j] -= Network.learningRate * weightGradSum[i][j] / samples.size();
            }
            layer.biases[i] -= Network.learningRate * biasGradSum[i] / samples.size();
        }
        return loss / samples.size();
    }
    
}
