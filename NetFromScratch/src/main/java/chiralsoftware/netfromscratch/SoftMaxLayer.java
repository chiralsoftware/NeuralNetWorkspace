package chiralsoftware.netfromscratch;

/** 
 f(z_i) = exp(z_i) / sum(exp(z_j)) for all j in the layer
*/
final class SoftMaxLayer extends Layer {

    SoftMaxLayer(Neuron[] neurons) {
        super(neurons);
    }

    @Override
    public float[] forward(float[] input) {
        float[] raw = raw(input);
        return softmax(raw);
    }

    @Override
    public float computeLoss(float[] prediction, float[] target) {
        for (int i = 0; i < prediction.length; i++) {
            if (target[i] == 1f) {
                return - (float) Math.log(prediction[i] + 1e-7f);
            }
        }
        return 0f;
    }

    @Override
    public float[] computeGradient(float[] prediction, float[] target) {
        float[] gradient = new float[prediction.length];
        for (int i = 0; i < prediction.length; i++) {
            gradient[i] = prediction[i] - target[i];
        }
        return gradient;
    }

    @Override
    public void update(float[] gradient, float[] input) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].adjust(gradient[i], input);
        }
    }

    private static float[] softmax(float[] raw) {
        float max = Float.NEGATIVE_INFINITY;
        for (float val : raw) {
            if (val > max) max = val;
        }   

        final float[] exp = new float[raw.length];
        float sum = 0f;
        for (int i = 0; i < raw.length; i++) {
            exp[i] = (float) Math.exp(raw[i] - max); // stability trick
            sum += exp[i];
        }

        for (int i = 0; i < raw.length; i++) {
            raw[i] = exp[i] / sum;
        }

        return raw;
    }

}
