import java.util.Random;

/*
* The objective of this project was to create a forward feed artificial neural network that utilizes the
* back propagation algorithm. The problem that we attempt to solve is the 4 bit parity problem.
* By inputting a binary number we add a parity bit in order to make the output an odd number of ones.
*
*
* */

public class NeuralNetwork {
    //initialize arrays
    Random rand = new Random(1);
    float[][] weight2 = new float[8][4];
    float[][] delta2 = new float[8][4];
    float[][] weight3 = new float[1][8];
    float[][] delta3 = new float[1][8];
    float[][] testingSet = new float[4][4];
    Neuron[] X1, X2, X3;
    float[] A1, A2, A3, y;
    //initialize variables
    float cost, totalCost = 0;
    //set adjustable learning rate to 15%
    float learningRate = (float) 0.15;


    public NeuralNetwork() {

        int epochs = 10000;
        String[] binary = initializeBinary();
        fixBinary(binary);
        float[][] inputArray = new float[12][4];
        randomizeIndex(inputArray, binary);
        y = new float[inputArray.length];
        calcExpected(inputArray);
        initializeWeight(weight2);
        initializeWeight(weight3);
        initializeArrays(inputArray[0]);
        calcGradients(0);
        applyLearningRate();
        updateWeights();


        for (int i = 1; i <= epochs; i++) {
            for (int j = 0; j < inputArray.length; j++) {
                forwardPass(inputArray[j]);
                calcGradients(j);
                applyLearningRate();
                updateWeights();
                totalCost += Math.pow(cost, 2);
            }
            if (i % 200 == 0) {
                System.out.println("Epoch: " + i + " | MSE: " + totalCost / i);
            }
        }
        finishedTraining(epochs);
        testSet(inputArray);
        float[][] testSet = setTestSet();
        testSet(testSet);
    }

    //this method creates a new float array with the inputs not used
    public float[][] setTestSet(){
        float[][] testSet = new float[4][4];
        testSet[0][0] = 1;
        testSet[0][1] = 0;
        testSet[0][2] = 1;
        testSet[0][3] = 0;

        testSet[1][0] = 1;
        testSet[1][1] = 0;
        testSet[1][2] = 0;
        testSet[1][3] = 1;

        testSet[2][0] = 1;
        testSet[2][1] = 0;
        testSet[2][2] = 0;
        testSet[2][3] = 0;

        testSet[3][0] = 0;
        testSet[3][1] = 1;
        testSet[3][2] = 1;
        testSet[3][3] = 1;

        return testSet;
    }

    //this method tests the input and outputs the number of correct guesses and the percentage of accuracu
    public void testSet(float[][] inputArray){
        int correctGuess = 0;
        for (int i = 0; i < inputArray.length; i++) {
            forwardPass(inputArray[i]);
            updateCost(i);
            output(i);
            correctGuess += countCorrectGuesses(correctGuess, i);
        }
        System.out.println("The ANN made " + correctGuess + " correct guesses.");
        float percentage = ((float)correctGuess / inputArray.length);
        System.out.println("This is " + percentage + " percent accurate");

    }

    //this method counts the amount of correct guesses in a set. This method is used in the testSet method
    public int countCorrectGuesses(int correctGuess, int inputArrayIndex){
        int guess;
        if (A3[0] > 0.49) {
            guess = 1;
        } else {
            guess = 0;
        }
        if (guess == y[inputArrayIndex]) {
            return 1;
        }
        return 0;
    }

    //this method outputs information on the training when the training is complete
    public void finishedTraining(int epochs) {
        System.out.println("--------------------------------------------------------------");
        System.out.println("Training complete!");
        System.out.println("Hidden nodes used: " + X2.length);
        System.out.println("Learning rate: " + learningRate);
        System.out.println("Final Mean Squared Error: " + totalCost / epochs);
    }

    //this method also outputs information on the training when the training is complete
    public void output(int inputArrayIndex) {
        System.out.print("Input: " + "[" + X1[0].getValue() + "," + X1[1].getValue() + "," + X1[2].getValue() + "," + X1[3].getValue() + "]" + " | ");
        int guess;
        if (A3[0] > 0.49) {
            guess = 1;
        } else {
            guess = 0;
        }
        System.out.print("Guess: " + guess + " | ");
        System.out.print("Expected: " + y[inputArrayIndex] + " | ");
        System.out.println("SE: " + Math.pow(cost, 2));

    }


    //This method handles the forward pass of the ANN. It receives a 1d array of input
    //and then calculates the sigmoid output and the sum*weight using the neuron class
    public void forwardPass(float[] input) {
        for (int i = 0; i < A1.length; i++) {
            Neuron n = new Neuron(input[i]);
            X1[i] = n;
        }
        calcSigmoidInput(X1, A1);
        //for each sigmoid input apply the weight matrix and sigmoid output as array of neurons
        for (int i = 0; i < weight2.length; i++) {
            A2[i] = X2[i].output(A1);
        }
        A3[0] = X3[0].output(A2);
    }


    //This method applies the learning rate to the weight matrices.
    public void applyLearningRate() {
        for (int i = 0; i < delta2.length; i++) {
            for (int j = 0; j < delta2[i].length; j++) {
                weight2[i][j] = weight2[i][j] - learningRate * delta2[i][j];
            }
        }

        for (int i = 0; i < delta3.length; i++) {
            weight3[0][i] = weight3[0][i] - learningRate * delta3[0][i];
        }

    }

    //this method updates the weights in the neuron class held by the arrays X2, X3
    public void updateWeights() {
        for (int i = 0; i < X2.length; i++) {
            X2[i].setWeight(weight2[i]);
        }
        for (int i = 0; i < X3.length; i++) {
            X3[i].setWeight(weight3[i]);
        }


    }

    //updates the costs of the guess. Also know as error
    public void updateCost(int inputArrayIndex) {
        //calc cost (A^3 - y)
        cost = A3[0] - y[inputArrayIndex];
    }

    //This method calculates the gradients by element multiplication and element wise multiplication.

    public void calcGradients(int inputArrayIndex) {

        updateCost(inputArrayIndex);
        float twoCost = 2 * cost;
        float sigmaX3Prime = 0;

        //calc sigma X^3 prime A^3 * (1 - A^3)
        for (int i = 0; i < A3.length; i++) {
            sigmaX3Prime = A3[i] * (1 - A3[i]);
        }

        //calc delta3 Scholar 2(A^3 - y) * Sigma X^3 prime
        float delta3Scholar = twoCost * sigmaX3Prime;

        //calculate the rest of delta3 (A^2 * delta3Scholar)
        for (int i = 0; i < A2.length; i++) {
            delta3[0][i] = A2[i] * delta3Scholar;
        }

        //calc first half of delta2 2(A^3 - y) * (A^3(1 - A^3)) * W^3
        float[] firstHalfOfDelta2 = new float[weight3[0].length];
        for (int i = 0; i < firstHalfOfDelta2.length; i++) {
            firstHalfOfDelta2[i] = delta3Scholar * weight3[0][i];
        }


        //calc sigma X2 A^2(1 - A^2)
        float[] sigmaX2Prime = new float[A2.length];
        for (int i = 0; i < A2.length; i++) {
            sigmaX2Prime[i] = A2[i] * (1 - A2[i]);
        }

        //calc 2/3's of delta3 (2(A^3 - y) * sigmaX^3 * W^3 * SigmaX^2
        for (int i = 0; i < A2.length; i++) {
            firstHalfOfDelta2[i] = firstHalfOfDelta2[i] * sigmaX2Prime[i];
        }

        //finish the calc of delta3
        for (int i = 0; i < firstHalfOfDelta2.length; i++) {
            for (int j = 0; j < A1.length; j++) {
                delta2[i][j] = firstHalfOfDelta2[i] * A1[j];
            }
        }


    }

    //This method randomizes an index for the input array. It will get a random integer, start at
    //that integer and continue until either the input array is filled or the end of the binary
    //array has been hit. If the end of the binary array has been reached, it continues with the
    //beginning of the array until the input array has been filled. This will randomize a selection
    //of inputs based on seed given.
    public void randomizeIndex(float[][] inputArray, String[] binary) {
        int index = rand.nextInt(binary.length);
        int arrayIndexing = 0;
        for (int i = index; i < binary.length; i++) {
            String copyFrom = binary[i];
            char[] inputCharArray = copyFrom.toCharArray();
            for (int j = 0; j < 4; j++) {
                inputArray[arrayIndexing][j] = Character.getNumericValue(inputCharArray[j]);
            }
            arrayIndexing++;
            if (arrayIndexing == inputArray.length) {
                return;
            }
        }
        int index2 = (inputArray.length - arrayIndexing);
        for (int i = 0; i < index2; i++) {
            String copyFrom = binary[i];
            char[] inputCharArray = copyFrom.toCharArray();
            for (int j = 0; j < 4; j++) {
                inputArray[arrayIndexing][j] = Character.getNumericValue(inputCharArray[j]);
            }
            arrayIndexing++;
        }
        arrayIndexing = 0;
        for (int i = index2; i < index; i++) {
            String copyFrom = binary[i];
            char[] inputCharArray = copyFrom.toCharArray();
            for (int j = 0; j < 4; j++) {
                testingSet[arrayIndexing][j] = Character.getNumericValue(inputCharArray[j]);
            }
            arrayIndexing++;
        }


    }

    //This method initializes the binary array by filling the array by counting to it's amount of
    //input possibilities in this case 16 and then turing that into binary.
    public String[] initializeBinary() {

        String[] binary = new String[16];
        for (int i = 0; i < 16; i++) {
            binary[i] = Integer.toBinaryString(i);
        }
        return binary;
    }

    //This function fixes the output from initializeBinary() by appending 0's
    public void fixBinary(String[] binary) {

        String length = binary[binary.length - 1];
        int binaryLength = length.length();

        for (int i = 0; i < binary.length; i++) {
            if (binary[i].length() != binaryLength) {
                for (int j = 0; j < binaryLength; j++) {
                    String zeros = "";
                    for (int k = 0; k < binaryLength - binary[i].length(); k++) {
                        zeros += "0";
                    }
                    binary[i] = zeros + binary[i];
                }
            }
        }
    }

    //This function initializes the weight arrays by filling it with random floats and multiplying
    //it randomly by -1 to set it to a random negative/positive float.
    public void initializeWeight(float[][] weight) {

        boolean negative;
        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < weight[i].length; j++) {
                negative = rand.nextBoolean();
                weight[i][j] = rand.nextFloat();
                if (negative) weight[i][j] *= -1;
            }
        }
    }

    //This function initializes the arrays and sets up the neuron arrays. This function does the
    //first forward pass through and gradient calculation and then after that it's handled in a
    //repeated function
    public void initializeArrays(float[] input) {
        X1 = new Neuron[weight2[0].length];
        A1 = new float[X1.length];
        X2 = new Neuron[weight2.length];
        A2 = new float[weight2.length];
        X3 = new Neuron[weight3.length];
        A3 = new float[weight3.length];

        for (int i = 0; i < A1.length; i++) {
            Neuron n = new Neuron(input[i]);
            X1[i] = n;
        }
        calcSigmoidInput(X1, A1);
        for (int i = 0; i < weight2.length; i++) {
            Neuron n = new Neuron(weight2[i]);
            X2[i] = n;
            A2[i] = n.output(A1);

        }
        Neuron n = new Neuron(weight3[0]);
        X3[0] = n;
        A3[0] = n.output(A2);
    }


    //This function calculates the sigmoid output of any neuron input and returns and sigmoidOutput
    //array
    public void calcSigmoidInput(Neuron[] input, float[] sigmoidOutput) {
        for (int i = 0; i < input.length; i++) {
            sigmoidOutput[i] = (1 / ((float) (1 + (Math.exp(-(input[i].getValue()))))));
        }
    }



    //This function calculates the corrected answer from any given input. This updates the array y
    //which is used to calculate error, SE, and MSE
    public void calcExpected(float[][] inputArray) {
        int count = 0;
        for (int i = 0; i < inputArray.length; i++) {
            for (int j = 0; j < inputArray[0].length; j++) {
                if (inputArray[i][j] == 1) count++;
            }
            if (count % 2 == 0) {
                y[i] = 1;
            } else {
                y[i] = 0;
            }
            count = 0;
        }
    }


    public static void main(String[] args) {
        NeuralNetwork ann = new NeuralNetwork();
    }

}
