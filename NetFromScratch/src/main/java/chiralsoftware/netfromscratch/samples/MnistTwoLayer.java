package chiralsoftware.netfromscratch.samples;

import chiralsoftware.netfromscratch.Image;
import chiralsoftware.netfromscratch.Layer;
import chiralsoftware.netfromscratch.MnistImageReader;
import chiralsoftware.netfromscratch.Network;
import chiralsoftware.netfromscratch.Sample;
import chiralsoftware.netfromscratch.SigmoidActivationLayer;
import chiralsoftware.netfromscratch.SoftMaxLayer;
import chiralsoftware.netfromscratch.TrainingTracker;
import com.google.common.collect.ImmutableList;
import java.io.IOException;
import static java.lang.Math.round;
import java.util.ArrayList;
import java.util.Arrays;
import static java.util.Collections.shuffle;
import java.util.List;
import java.util.Random;

/** Train a basic two-layer MNIST network that will get to reasonable accuracy very
 quickly */
public final class MnistTwoLayer {
    
    private final ArrayList<Layer> layers = new ArrayList();
    private final ArrayList<Sample> trainSamples;
    
    private Network network;
    
    public List<Layer> getLayers() {
        return ImmutableList.copyOf(layers);
    }
    
    public MnistTwoLayer(TrainingTracker trainingTracker) throws IOException {
        MnistImageReader.load();
        
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

        network = new Network(layers, trainingTracker);
        
        List<Image> collection = Arrays.asList(MnistImageReader.images);
        shuffle(collection);
        collection = collection.subList(0, 4000); // trim the training set size
        
        final List<Sample> samples = new ArrayList<>();
        
        for(int i = 0; i < collection.size(); i++) {
            final Image image = collection.get(i);
            final float[] imageFloat = new float[28*28];
            image.toFloat(imageFloat);
            final float[] target = new float[10];
            target[image.label()] = 1f;
            samples.add(new Sample<Image>(imageFloat, target, image));
        }
        
        final Random random = new Random();
        shuffle(samples);
        final int splitPoint = (int) round(samples.size() * 0.8);
        trainSamples = new ArrayList(splitPoint);
        for(int i = 0; i < splitPoint; i++) {
            trainSamples.add(samples.get(i));
        }
        final ArrayList<Sample> testSamples = new ArrayList(samples.size() - trainSamples.size());
        for(int i = 0; i < samples.size() - splitPoint; i++)
            testSamples.add(samples.get(i + trainSamples.size()));
    }
    
    public void doIt() throws IOException {
        System.out.println("ok i shoudl START TRAINING");
        network.train(trainSamples);
    }
    
}
