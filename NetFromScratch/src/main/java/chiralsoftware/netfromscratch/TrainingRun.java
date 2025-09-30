package chiralsoftware.netfromscratch;

import static java.lang.Math.round;
import static java.lang.System.Logger.Level.INFO;
import java.util.ArrayList;
import java.util.List;

/**
 * Perform a training run, which is a set of Epochs
 */
final class TrainingRun {

    private static final System.Logger LOG = System.getLogger(TrainingRun.class.getName());
    
    static final float trainingSplit = 0.8f;
    
    TrainingRun(Sample[] allSamples, Network network) {
        // 1. split the samples into Train and Test
        final List<Sample> training = new ArrayList<>();
        final List<Sample> testing = new ArrayList<>();
        final int trainingSize = round(allSamples.length * trainingSplit);
        final int testSize = allSamples.length - trainingSize;
        for(int i = 0; i < trainingSize; i++) training.add(allSamples[i]);
        for(int i = trainingSize; i < allSamples.length; i++)
            testing.add(allSamples[i]);
        LOG.log(INFO, "split " + allSamples.length + 
                " into " + training.size() + 
                " training samples and " + testing.size() + " testing samples");
        // 2. loop for number of epochs:
        //    1. divide training set into batches
        //    2. loop over batches and update weights after each batch
        // 3. evaluate on test set
    }
    
}
