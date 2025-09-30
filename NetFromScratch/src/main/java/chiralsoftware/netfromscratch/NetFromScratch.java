package chiralsoftware.netfromscratch;

import static java.lang.System.out;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

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
        
        for(int i = 101; i < 1150; i++) {
            final Image image = MnistImageReader.images[i];
            out.println("Looking at image: " + image.label());
            out.println(image.show());
            final float[] imageFloat = new float[28*28];
            image.toFloat(imageFloat);
            final float[] target = new float[10];
            target[image.label()] = 1f;
            samples.add(new Sample(imageFloat, target));
        }

        final Random random = new Random();
        
        network.train(samples);
        out.println("Training complete. Doing some predictions....");
        for(int i = 0; i < 10; i++) {
            final Sample sample = samples.get(random.nextInt(samples.size()));
            final float[] prediction = network.predict(sample.x());
            out.println(sample.toString());
            out.println(Network.showTarget(sample.target(), prediction));
            out.println();
            
        }
    }
}
