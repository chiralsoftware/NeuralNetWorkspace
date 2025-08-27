package chiralsoftware.netfromscratch;

import static chiralsoftware.netfromscratch.Activations.sigmoidLogistic;
import static chiralsoftware.netfromscratch.Losses.mseLoss;
import static java.lang.System.out;
import java.lang.annotation.Target;
import java.text.DecimalFormat;
import java.util.Arrays;

public final class NetFromScratch {

    public static void main(String[] args) throws Exception {
        out.println("Testing network from scratch");
        MnistImageReader.load();
        out.print("Mnist set loaded");
        
        out.println("creating ten neurons");
        final Neuron[] neurons = new Neuron[10];
        for(int i = 0; i < neurons.length; i++ ) {
            neurons[i] = Neuron.random(28*28);
        }
        out.print("creating the one layer");

        final Image image = MnistImageReader.images[101];
        out.println("Looking at image: " + image.label());
        out.println(image.show());
        
        out.println("applying network");
        
        final Layer layer = new SoftMaxLayer(neurons);
        final Network network = new Network(layer);
        
        final float[] imageFloat = new float[28*28];
        image.toFloat(imageFloat);
        

        final float[] output = network.predict(imageFloat);
        out.println("and the output is: "+ Arrays.toString(output));
        final float[] target = new float[10];
        target[image.label()] = 1;
        
        network.train(imageFloat, target);
        
    }
}
