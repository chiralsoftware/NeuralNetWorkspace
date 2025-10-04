package chiralsoftware.netfromscratch;

import static java.lang.Math.round;
import static java.lang.System.out;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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

        final Layer layer = new SoftMaxLayer(neurons);
//        final Layer layer = new ActivationLayer(neurons,Activations.sigmoidLogistic, Losses.mseLoss());
        final Network network = new Network(layer);
        
        final Image[] sublist = new Image[(int) round(MnistImageReader.images.length * 0.2)];
        List<Image> subCollection = Arrays.asList(MnistImageReader.images);
        Collections.shuffle(subCollection);
        subCollection = subCollection.subList(0, 10000);
        
        final List<Sample> samples = new ArrayList<>();
        
        out.println("applying network");
        for(int i = 0; i < subCollection.size(); i++) {
            final Image image = subCollection.get(i);
            if(i % 1000 == 0) {
                out.println("Looking at image: " + image.label());
                out.println(image.show());
            }
            final float[] imageFloat = new float[28*28];
            image.toFloat(imageFloat);
            final float[] target = new float[10];
            target[image.label()] = 1f;
            samples.add(new Sample(imageFloat, target));
        }
        
        final Random random = new Random();
        Collections.shuffle(samples);
        final int splitPoint = (int) round(samples.size() * 0.8);
        final ArrayList<Sample> trainSamples = new ArrayList(splitPoint);
        for(int i = 0; i < splitPoint; i++) {
            trainSamples.add(samples.get(i));
        }
        final ArrayList<Sample> testSamples = new ArrayList(samples.size() - trainSamples.size());
        for(int i = 0; i < samples.size() - splitPoint; i++) {
            testSamples.add(samples.get(i + trainSamples.size()));
        }
        network.train(trainSamples);
        out.println("Training complete. Doing some predictions....");
        for(int i = 0; i < 10; i++) {
            final Sample sample = testSamples.get(random.nextInt(testSamples.size()));
            final float[] prediction = network.predict(sample.x());
            out.println(sample.toString());
            out.println(Network.showTarget(sample.target(), prediction));
            out.println();
            
        }
    }
}
