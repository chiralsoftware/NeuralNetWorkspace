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
    
    private static final float learningRate = 0.05f;
    
    static void train(Image[] images) {
        // now we need to adjust these weights
    }
    
}
