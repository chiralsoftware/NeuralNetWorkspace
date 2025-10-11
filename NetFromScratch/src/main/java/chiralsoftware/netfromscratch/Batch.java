package chiralsoftware.netfromscratch;

import java.util.ArrayList;
import java.util.List;

/**
 * Represent a batch of samples and perform training and update within the batch
 */
final class Batch {
    
    private final List<Sample> samples;
    private final ArrayList<Layer> layers;

    Batch(List<Sample> samples, ArrayList<Layer> layers) {
        this.samples = samples;
        this.layers = layers;
    }
    
    float process() {
        final Layer layer = layers.get(0); // only one layer for now
        final float[][] weightGradSum = new float[layer.outputSize][layer.inputSize];
        final float[] biasGradSum = new float[layer.outputSize];
        float loss = 0;
        for(int sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++) {
            final Sample sample = samples.get(sampleIndex);
            final float[] grad = layer.gradient(sample.x(), sample.target());
            loss += layer.loss(sample.x(), sample.target());
            for (int i = 0; i < layer.outputSize; i++) {
                for (int j = 0; j < layer.inputSize; j++) {
                    weightGradSum[i][j] += grad[i] * sample.x()[j];
                }
                biasGradSum[i] += grad[i];
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
