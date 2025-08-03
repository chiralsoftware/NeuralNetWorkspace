package chiralsoftware.netfromscratch;

import static chiralsoftware.netfromscratch.Activations.sigmoidLogistic;
import static java.lang.System.out;
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
        
        final Layer layer = new Layer(neurons, sigmoidLogistic);
        
        out.println("Apply the layer to a test image");
        final Image image = MnistImageReader.images[101];
        out.println("Looking at image: " + image.label());
        out.println(image.show());
        
        out.println("applying network");
        
        final float[] imageFloat = new float[28*28];
        image.toFloat(imageFloat);
        final float[] activated = layer.calculate(imageFloat);
        out.println("and the amazing result is: "+ Arrays.toString(activated));
        final float[] target = new float[10];
        target[image.label()] = 1f;
        
        out.println("calculating error:");
        float error = 0f;
        for (int i = 0; i < 10; i++) {
            float diff = activated[i] - target[i];
            error += diff * diff;
        }
        error /= 10f;
        out.println("MSE error: " + error);

    }
}
