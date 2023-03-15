package chiralsoftware.netfromscratch;

/**
 * Most basic learning task - binary classification, is this digit a 7 or not?
 */
public final class PrimitiveLearning {
    
    static void adjustNeuron(Image training, Neuron neuron) {
        final float result = neuron.calculate(training.data());
        final float target = training.label() == 7 ? 1 : 0;
    }
    
}
