package chiralsoftware.netfromscratch;

/**
 * Loss functions
 */
final class Losses {
    
    static LossFunction mseLoss() {
        return new LossFunction() {
            @Override
            public float loss(float y_hat, float y) {
                return (y_hat - y) * (y_hat -y);
            }

            @Override
            public float derivative(float y_hat, float y) {
                return -2 * (y_hat - y);
            }
        };
    }

}
