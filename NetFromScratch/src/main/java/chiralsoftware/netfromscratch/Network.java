package chiralsoftware.netfromscratch;

import static java.lang.System.Logger.Level.INFO;
import java.util.Arrays;

final class Network {

    private static final System.Logger LOG = System.getLogger(Network.class.getName());

    private final Layer outputLayer;
    
    static final float learningRate = 0.05f;
    
    public Network(Layer outputLayer) {
        this.outputLayer = outputLayer;
    }
    
    float[] predict(float[] x) {
        return outputLayer.forward(x);
    }
    
    int epochs = 100000;
    
    void train(float[] x, float[] target) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            final float[] prediction = outputLayer.forward(x);
            final float loss = outputLayer.computeLoss(prediction, target);
            final float[] gradient = outputLayer.computeGradient(prediction, target);
            outputLayer.update(gradient, x);

            if (epoch % 100 == 0) {
                LOG.log(INFO, "Epoch " + epoch + " Loss: " + loss);
                LOG.log(INFO, "Prediction: " + Arrays.toString(prediction));
            }
        }
    }

}
