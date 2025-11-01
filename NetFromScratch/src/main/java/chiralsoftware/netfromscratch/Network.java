package chiralsoftware.netfromscratch;

import static com.google.common.collect.Lists.partition;
import static java.lang.System.Logger.Level.INFO;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import static java.util.stream.Collectors.toUnmodifiableList;

final class Network {

    private static final System.Logger LOG = System.getLogger(Network.class.getName());

    /** 
     * Layer 0 is the first real layer with weights and activations. The Nth layer
     * is the output layer.
     */
    private final ArrayList<Layer> layers;
    
    static final float learningRate = 0.05f;
    
    public Network(ArrayList<Layer> layers) {
        this.layers = layers;
    }
    
    float[] predict(float[] x) {
        float[] output = x;
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            final Layer layer = layers.get(layerIndex);
            output = layer.activated(layer.raw(output));
        }
        return output;
    }
    
    int epochs = 200;
    
    void train(List<Sample> samples) {
        final DecimalFormat df = new DecimalFormat("0.0000");
        final List<List<Sample>> partitionedSamples = partition(samples, 32);
        final List<Batch> batches = partitionedSamples.stream().
                map(listOfSamples -> new Batch(listOfSamples, layers)).
                collect(toUnmodifiableList());
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            float loss = 0;
            for(Batch b : batches) {
                b.process();
            }
            if (epoch % 50 == 0) {
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
