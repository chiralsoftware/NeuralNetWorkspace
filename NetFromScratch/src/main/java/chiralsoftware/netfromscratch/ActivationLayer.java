package chiralsoftware.netfromscratch;

/**
 * Mean Square Error layer
 */
abstract class ActivationLayer extends Layer {

    private ActivationFunction activationFunction;
    private LossFunction lossFunction;
    
    ActivationLayer(int inputSize, int outputSize) {
        super(inputSize, outputSize);
    }

    float computeLoss(float[] prediction, float[] target) {
        float sum = 0;
        for(int i = 0; i < prediction.length; i++) {
            sum += lossFunction.loss(target[i], prediction[i]);
        }
        return sum / target.length;
    }


    @Override
    void update(float[] accumulatedWeightedGradient, float[]  accumulatedBiasGradient) {
    }

    @Override
    protected float[] activated(float[] raw) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    @Override
    protected float loss(float[] input, float[] target) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }
    
}
