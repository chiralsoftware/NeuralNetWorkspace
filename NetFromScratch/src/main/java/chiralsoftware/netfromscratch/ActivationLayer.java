package chiralsoftware.netfromscratch;

/**
 * Mean Square Error layer
 */
final class ActivationLayer extends Layer {

    private final ActivationFunction activationFunction;
    private final LossFunction lossFunction;
    
    ActivationLayer(Neuron[] neurons, ActivationFunction activationFunction, LossFunction lossFunction) {
        super(neurons);
        this.activationFunction = activationFunction;
        this.lossFunction = lossFunction;
    }

    @Override
    float[] forward(float[] input) {
        final float raw[] = raw(input);
        for(int i = 0; i < raw.length; i++)
            raw[i] = activationFunction.activate(raw[i]);
        return raw;
    }

    @Override
    float computeLoss(float[] prediction, float[] target) {
        float sum = 0;
        for(int i = 0; i < prediction.length; i++) {
            sum += lossFunction.loss(target[i], prediction[i]);
        }
        return sum / target.length;
    }

    @Override
    float[] computeGradient(float[] prediction, float[] target) {
        final float[] result = new float[prediction.length];
        for(int i = 0; i < result.length; i++) 
            result[i] = lossFunction.derivative(target[i], prediction[i]);
        return result;
    }

    @Override
    void update(float[] gradient, float[] input) {
        for(int i =0 ;i < neurons.length; i++)
            neurons[i].adjust(gradient[i], input);
    }
    
}
