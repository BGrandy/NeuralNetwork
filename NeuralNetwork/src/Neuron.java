
public class Neuron {

    float[] weights;
    float value;

	//constructors
    public Neuron(float[] weight) {
        weights = weight;

    }
    public Neuron(float value){
        this.value = value;
    }


    public float getValue() {
        return value;
    }

    public void setWeight(float[] weight) {
        weights = weight;
    }

    //Output the sum which is the weight at i and the input at i multiplied together and then
    //added. Then send that sum through the activation function and return that output.
    public float output(float[] inputs) {
        float sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return activation(sum);
    }

    public float activation(float sum) {
        return 1 / ((float) (1 + Math.exp(-(sum))));
    }



}// Neuron class
