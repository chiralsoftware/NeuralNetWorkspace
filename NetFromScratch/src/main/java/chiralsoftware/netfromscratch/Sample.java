package chiralsoftware.netfromscratch;

/**
 * Represent a single training simple with input x and expected target
 */
public record Sample<T>(float[] x, float[] target, T data) {
    
}
