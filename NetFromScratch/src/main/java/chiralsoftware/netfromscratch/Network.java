package chiralsoftware.netfromscratch;

import static java.lang.System.Logger.Level.INFO;
import java.util.Arrays;

final class Network {

    private static final System.Logger LOG = System.getLogger(Network.class.getName());

    private final Layer outputLayer;
    private final LossFunction lossFunction;
    
    public Network(Layer outputLayer, LossFunction loss) {
        this.outputLayer = outputLayer;
        this.lossFunction = loss;
    }
    
    float[] predict(float[] x) {
        return outputLayer.forward(x);
    }
    
    void train(float[] x, float[] target) {
        LOG.log(INFO, "starting to train for target: " + Arrays.toString(target));
        final float[] result1 = predict(x);
        final float[] raw1 = outputLayer.rawOutputs(x);

        float error = 0f;
        for (int i = 0; i < 10; i++) {
            error += lossFunction.loss(target[i], result1[i]);
        }
        error /= 10f;
        LOG.log(INFO, "MSE error: " + error);
        
        final float[] lossDerivative = new float[10];
        for(int i = 0; i < raw1.length ; i++) 
            lossDerivative[i] = lossFunction.derivative(target[i], raw1[i]);
        final float[] activationDerivative = new float[raw1.length];
        for(int i = 0; i < raw1.length; i++) 
            activationDerivative[i] = Activations.sigmoidLogistic.derivative(raw1[i]);
        LOG.log(INFO, "loss derivative and activation derivative calculated");
        final float[] gradients = new float[raw1.length];
        for(int i = 0; i < raw1.length; i++)
            gradients[i] = activationDerivative[i] * lossDerivative[i];
        outputLayer.update(gradients, x);
        
        LOG.log(INFO, "now that update has been done calculate the new error");
        final float[] prediction2 = predict(x);
        float error2 = 0f;
        for (int i = 0; i < 10; i++) {
            error2 += lossFunction.loss(target[i], prediction2[i]);
        }
        error2 /= 10f;
        LOG.log(INFO, "previous error: " + error);
        LOG.log(INFO, "new error: " + error2);

    }
}
