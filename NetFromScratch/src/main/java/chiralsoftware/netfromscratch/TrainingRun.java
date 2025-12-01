package chiralsoftware.netfromscratch;

import java.io.IOException;
import static java.lang.Math.round;
import static java.lang.System.Logger.Level.INFO;
import java.util.ArrayList;
import static java.util.Collections.shuffle;
import java.util.List;

/**
 * Perform a training run, which is a set of Epochs
 */
public final class TrainingRun {

    private static final System.Logger LOG = System.getLogger(TrainingRun.class.getName());
    
    static final float trainingSplit = 0.8f;
    private final List<Sample> training;
    private final List<Sample> testing;
    private final Network network;
    
    private TrainingTracker trainingTracker = null;
    
    public TrainingRun(ArrayList<Sample> allSamples, Network network) {
        shuffle(allSamples);
        // split the samples into Train and Test
        this.network = network;
        final int trainingSize = round(allSamples.size() * trainingSplit);
        training = allSamples.subList(0, trainingSize);
        testing = allSamples.subList(trainingSize, allSamples.size());
        LOG.log(INFO, "split " + allSamples.size() + 
                " into " + training.size() + 
                " training samples and " + testing.size() + " testing samples");
    }
    
    public void setTrainingTracker(TrainingTracker trainingTracker) {
        this.trainingTracker = trainingTracker;
    }
    
    public void start() {
        if(trainingTracker != null) trainingTracker.setStartTime();
        network.train(training);
    }

}
