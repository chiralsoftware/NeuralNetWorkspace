package chiralsoftware.netfromscratch;

import static chiralsoftware.netfromscratch.Activations.sigmoidLogistic;
import static java.lang.System.out;

public final class NetFromScratch {

    public static void main(String[] args) throws Exception {
        out.println("Testing network from scratch");
        
        out.println("Step 1: create a single neuron of the right shape and initialize it");
        
        final Neuron neuron = new Neuron(28*28, sigmoidLogistic);
        neuron.initialize();
        out.println("Created a neuron: " + neuron);
        out.println("Step 2: load the MINST images");
        
        
        out.println("Step 3: apply the neuron to a test image");
        
        final float result = neuron.calculate(MnistImageReader.images[69].data());
        out.println("The result: " + result);
    }
}
