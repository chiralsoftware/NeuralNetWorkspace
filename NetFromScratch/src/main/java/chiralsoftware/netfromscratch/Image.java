package chiralsoftware.netfromscratch;

/**
 * Hold a sample MNIST image, which includes the pixels and the label
 */
public record Image(int label, byte[] data) {
    
    /** Output this image with label */
    public String show() {
        if(data.length != 28*28 ) return "(invalid data length: " + data.length + ")";
        final StringBuilder result = new StringBuilder();
        for(int y = 0; y < 28; y++) {
            for(int x = 0; x < 28; x++) {
                if(x == 0 && y == 0)
                    result.append(Integer.toString(label));
                else
                    result.append(imageByteToAscii(data[y*28+x]));
            }
            result.append("\n");
        }
        return result.toString();
    }
    
    void toFloat(float[] fa) {
        if(fa.length != data.length) throw new RuntimeException("float array length: " + fa.length + 
                " did not match data array length");
        for(int i = 0; i < data.length; i++ ) {
            fa[i] = (float) (data[i] & 0xff) / 256f;
        }
    }
    
    public static String imageByteToAscii(byte b) {
        final int value = b & 0xff;
        if(value == 0) return " ";
        if(value < 50) return "░";
        if(value < 100) return "▒";
        if(value < 200) return "▓";
        return "█";
    }
    
}
