package chiralsoftware.netfromscratch;

import static chiralsoftware.netfromscratch.Activations.sigmoidLogistic;
import java.util.Random;
import java.util.logging.Logger;

/**
 * Implement training. Start by building one single neuron,
 * and loop through training data and adjust.
 */
public final class Train {

    private static final Logger LOG = Logger.getLogger(Train.class.getName());
    
    private static final Random random = new Random();
    
    static void train(Image[] images) {
        final Neuron neuron = new Neuron(28 *28);
        final float[] fa = new float[28*28];
        
        neuron.initialize();
        LOG.info("The neuron right now: " + neuron);
        
        for(int i = 0; i < 10; i++ ) {
            // pick a representative image
            final Image im = images[random.nextInt(images.length)];
            im.toFloat(fa);
            final float result=neuron.calculate(fa);
            final float expectedResult;
            if(im.label() == 7) expectedResult = 1;
            else expectedResult = 0;
            final float error = expectedResult - result;
            
            LOG.info("iteration " + i + ", label: + " + im.label() + ", result: " + result + ", error=" + error);
        }
    }
    
}
