package chiralsoftware.netfromscratch;

import static chiralsoftware.netfromscratch.BatchProcessor.zero2dArray;
import static com.google.common.collect.Lists.partition;
import static java.lang.System.Logger.Level.INFO;
import static java.lang.System.arraycopy;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

final class Network {

    private static final System.Logger LOG = System.getLogger(Network.class.getName());

    /** 
     * Layer 0 is the first real layer with weights and activations. The Nth layer
     * is the output layer.
     */
    private final ArrayList<Layer> layers;
    
    static final float learningRate = 0.5f;
    
    public Network(ArrayList<Layer> layers) {
        this.layers = layers;
        rawOutputs = new float[layers.size()][];
        activatedOutputs = new float[layers.size()][];
        for(int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            rawOutputs[layerIndex] = new float[layers.get(layerIndex).outputSize];
            activatedOutputs[layerIndex] = new float[layers.get(layerIndex).outputSize];
        }
    }
    
    private final float[][] rawOutputs;
    private final float[][] activatedOutputs;
    
    void predict(float[] x, float[] result) {
        if(x.length != layers.getFirst().inputSize)
            throw new IllegalArgumentException("the network input size: "+ x.length + 
                    " did not equal the first layer input size: "+ layers.getFirst());
        
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            final Layer layer = layers.get(layerIndex);
            layer.raw(layerIndex == 0 ? x : activatedOutputs[layerIndex - 1], 
                    rawOutputs[layerIndex]);
             layer.activated(rawOutputs[layerIndex], activatedOutputs[layerIndex]);
        }
        arraycopy(activatedOutputs[activatedOutputs.length - 1], 0, result, 0, result.length);
    }
    
    int epochs = 200;
    
    void train(List<Sample> samples) {
        final DecimalFormat df = new DecimalFormat("0.00000000");
        final List<List<Sample>> partitionedSamples = partition(samples, 32);
        final BatchProcessor batchProcessor = new BatchProcessor(layers);
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            float loss = 0;
            for(List<Sample> batch : partitionedSamples) {
                loss = batchProcessor.process(batch);
            }
            if (epoch % 10 == 0) {
                LOG.log(INFO, "Epoch " + epoch + ", loss: " + df.format(loss));
            }            
        }
    }
    
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_RED = "\u001B[31m";
    private static final String ANSI_GREEN = "\u001B[32m";
    
    static String showTarget(float[] target, float[] prediction, Image image) {
        int correctResult = -1;
        float max = Float.MIN_VALUE;
        for(int i = 0; i < target.length; i++) {
            if(target[i] > max) {
                max = target[i];
                correctResult = i;
            }
        }
        if(correctResult < 0) return "ERROR1";
        int predicted = -1;
        max = Float.MIN_VALUE;
        for(int i = 0; i < prediction.length; i++) {
            if(prediction[i] > max) {
                max = prediction[i];
                predicted = i;
            }
        }
        if(predicted < 0) return "ERROR2";
        final boolean correct = correctResult == predicted;
        
        final DecimalFormat df = new DecimalFormat("0.0000");
        final StringBuilder sb = new StringBuilder();
        boolean resetRequired = false;
        for(int i = 0 ; i < prediction.length; i++ ) {
            if(! correct && i == predicted) {
                sb.append(ANSI_RED);
                resetRequired = true;
            }
            if(i == correctResult) {
                sb.append(ANSI_GREEN);
                resetRequired = true;
            }
            sb.append(df.format(prediction[i]));
            if(resetRequired) {
                sb.append(ANSI_RESET);
                resetRequired = false;
            }
            sb.append("   ");
        }
        if(! correct) {
            sb.append(image.show()).append("\n");
        }
        return sb.toString();
    }

}
