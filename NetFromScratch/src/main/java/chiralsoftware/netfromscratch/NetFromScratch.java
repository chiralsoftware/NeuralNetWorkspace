package chiralsoftware.netfromscratch;

import static java.lang.Math.round;
import static java.lang.System.out;
import java.util.ArrayList;
import java.util.Arrays;
import static java.util.Collections.shuffle;
import java.util.List;
import java.util.Random;

public final class NetFromScratch {

    public static void main(String[] args) throws Exception {
        out.println("Testing network from scratch");
        MnistImageReader.load();
        out.print("Mnist set loaded");
        out.print("creating the one layer");

//        final Layer layer = new SoftMaxLayer(28*28, 10);
//        final Layer layer = new SoftMaxLayer(28*28, 10);
//        layer.initRandom();
//        final ArrayList<Layer> layers = new ArrayList();
//        layers.add(layer);

        final ArrayList<Layer> layers = new ArrayList();
        
//        final SigmoidActivationLayer sal = new SigmoidActivationLayer(28*28, 64);
//        sal.initRandom();
//        layers.add(sal);
//        final SoftMaxLayer soft = new SoftMaxLayer(64, 10);
//        soft.initRandom();
//        layers.add(soft);
        
//        final SoftMaxLayer soft = new SoftMaxLayer(28*28, 10);
//        soft.initRandom();
//        layers.add(soft);
        
        final SigmoidActivationLayer sal1 = new SigmoidActivationLayer(28*28, 64);
        sal1.initRandom();
        layers.add(sal1);
        final SigmoidActivationLayer sal2 = new SigmoidActivationLayer(64,32);
        sal2.initRandom();
        layers.add(sal2);
        final SoftMaxLayer soft = new SoftMaxLayer(32, 10);
        soft.initRandom();
        layers.add(soft);

final Network network = new Network(layers);
        
        List<Image> collection = Arrays.asList(MnistImageReader.images);
        shuffle(collection);
        collection = collection.subList(0, 4000); // trim the training set size
        
        final List<Sample> samples = new ArrayList<>();
        
        out.println("applying network");
        for(int i = 0; i < collection.size(); i++) {
            final Image image = collection.get(i);
            if(i % 100 == 0) {
                out.println("Looking at image: " + image.label());
                out.println(image.show());
            }
            final float[] imageFloat = new float[28*28];
            image.toFloat(imageFloat);
            final float[] target = new float[10];
            target[image.label()] = 1f;
            samples.add(new Sample<Image>(imageFloat, target, image));
        }
        
        final Random random = new Random();
        shuffle(samples);
        final int splitPoint = (int) round(samples.size() * 0.8);
        final ArrayList<Sample> trainSamples = new ArrayList(splitPoint);
        for(int i = 0; i < splitPoint; i++) {
            trainSamples.add(samples.get(i));
        }
        final ArrayList<Sample> testSamples = new ArrayList(samples.size() - trainSamples.size());
        for(int i = 0; i < samples.size() - splitPoint; i++)
            testSamples.add(samples.get(i + trainSamples.size()));
        network.train(trainSamples);
        out.println("Training complete. Doing some predictions....");
        final float[] outputArray = new float[10];
        for(int i = 0; i < 10; i++) {
            final Sample<Image> sample = testSamples.get(random.nextInt(testSamples.size()));
            network.predict(sample.x(), outputArray);
            out.println(sample.toString());
            out.println(Network.showTarget(sample.target(), outputArray, sample.data()));
            out.println();
            
        }
    }
}
