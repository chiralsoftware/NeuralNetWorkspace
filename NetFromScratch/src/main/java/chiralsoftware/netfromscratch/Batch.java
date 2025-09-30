package chiralsoftware.netfromscratch;

import java.util.List;

/**
 * Represent a batch of samples and perform training and update within the batch
 */
final class Batch {
    
    private final List<Sample> samples;
    private final Layer layer;

    Batch(List<Sample> samples, Layer layer) {
        this.samples = samples;
        this.layer = layer;
    }
    
    float process() {
        final float[][] accumulatedWeightGradients = 
                new float[layer.neurons.length][samples.getFirst().x().length];
        final float[] accumulatedBiasGradients = new float[layer.neurons.length];

        float accumulatedLoss = 0;
        for(Sample s : samples) {
            final float[] prediction = layer.forward(s.x());
            // add in a computeLoss here so we can watch the loss
            final float[] delta = layer.computeGradient(prediction, s.target());
            for(int i = 0; i < layer.neurons.length; i++) {
                for(int j = 0; j < s.x().length; j++) {
                    accumulatedWeightGradients[i][j] += delta[i] * s.x()[j];
                }
                accumulatedBiasGradients[i] += delta[i]; 
            }
            accumulatedLoss += layer.computeLoss(prediction, s.target());
        }
        
        for(int i = 0 ;i < layer.neurons.length; i++) {
            for(int j= 0; j < accumulatedWeightGradients[i].length; j++)  {
                accumulatedWeightGradients[i][j] /= samples.size();
            }
            accumulatedBiasGradients[i] /= samples.size();                
        }

        accumulatedLoss /= samples.size();
        // now we have the accumulated gradient - adjust 
        for (int i = 0; i < layer.neurons.length; i++) {
            layer.neurons[i].adjust(accumulatedWeightGradients[i], accumulatedBiasGradients[i]);
        }
        return accumulatedLoss;
    }
    
}
