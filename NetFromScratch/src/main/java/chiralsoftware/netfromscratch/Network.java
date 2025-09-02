package chiralsoftware.netfromscratch;

import static java.lang.System.Logger.Level.INFO;
import java.text.DecimalFormat;
import java.util.Collection;

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
    
    int epochs = 50000;
    
    void train(Collection<Sample> samples) {
        final DecimalFormat df = new DecimalFormat("0.0000");
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (Sample sample : samples) {
                final float[] prediction = outputLayer.forward(sample.x());
                final float loss = outputLayer.computeLoss(prediction, sample.target());
                final float[] gradient = outputLayer.computeGradient(prediction, sample.target());
                outputLayer.update(gradient, sample.x());
                if (epoch % 10000 == 0) {
                    LOG.log(INFO, "Epoch " + epoch + " Loss: " + df.format(loss));
                    LOG.log(INFO, "Prediction: " + showTarget(sample.target(), prediction));
                }
            }
            
        }
    }
    
    private static final String ANSI_RESET = "\u001B[0m";
    private static final String ANSI_GREEN = "\u001B[32m";
    
    private static String showTarget(float[] target, float[] prediction) {
        final DecimalFormat df = new DecimalFormat("0.0000");
        final StringBuilder sb = new StringBuilder();
        for(int i = 0 ; i < prediction.length; i++ ) {
            final boolean yeahMan = target[i] > 0.99f;
            if(yeahMan) sb.append(ANSI_GREEN);
            sb.append(df.format(prediction[i]));
            if(yeahMan) sb.append(ANSI_RESET);
            sb.append("   ");
        }
        return sb.toString();
    }

}
