package chiralsoftware.netfromscratch;

/**
 *
 */
interface LossFunction {
    
    float loss(float y_hat, float y);
    
    float derivative(float y_hat, float y);
    
}
