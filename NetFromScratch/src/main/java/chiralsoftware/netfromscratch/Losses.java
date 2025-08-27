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
    
    /**
     loss = -[y * log(a) + (1 - y) * log(1 - a)]
    loss' = a - y
    */
    static LossFunction crossEntropyLoss = new LossFunction() {
        @Override
        public float loss(float y_hat, float y) {
            return  -1 * (
                    y_hat * (float) Math.log(y) + (1 - y_hat) * (float) Math.log(1 - y)
            );
        }

        @Override
        public float derivative(float y_hat, float y) {
            return y - y_hat;
        }
    };

}
