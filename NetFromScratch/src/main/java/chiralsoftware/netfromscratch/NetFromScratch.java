package chiralsoftware.netfromscratch;

import static java.lang.System.out;
import java.util.ArrayList;
import java.util.List;

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

//        final Layer layer = new SoftMaxLayer(neurons);
        final Layer layer = new ActivationLayer(neurons,Activations.sigmoidLogistic, Losses.mseLoss());
        final Network network = new Network(layer);
        
        final List<Sample> samples = new ArrayList<>();

        
        out.println("applying network");
        
        for(int i = 101; i < 150; i++) {
            final Image image = MnistImageReader.images[i];
            out.println("Looking at image: " + image.label());
            out.println(image.show());
            final float[] imageFloat = new float[28*28];
            image.toFloat(imageFloat);
            final float[] target = new float[10];
            target[image.label()] = 1f;
            samples.add(new Sample(imageFloat, target));
        }

        network.train(samples);
        
    }
}
