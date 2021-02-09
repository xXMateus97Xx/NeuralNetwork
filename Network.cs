namespace NeuralNetwork
{
    public class Network
    {
        private Matrix _biasIH, _biasHO, _weigthsIH, _weightsHO;
        private float _learningRate = 0.1f;

        public Network(int iNodes, int hNodes, int oNodes)
        {
            INodes = iNodes;
            HNodes = hNodes;
            ONodes = oNodes;

            _biasIH = new Matrix(hNodes, 1);
            _biasIH.Randomize();
            _biasHO = new Matrix(oNodes, 1);
            _biasHO.Randomize();

            _weigthsIH = new Matrix(hNodes, iNodes);
            _weigthsIH.Randomize();

            _weightsHO = new Matrix(oNodes, hNodes);
            _weightsHO.Randomize();
        }

        public int INodes { get; }
        public int HNodes { get; }
        public int ONodes { get; }

        public void Train(float[] arr, float[] target)
        {
            var input = new Matrix(arr);
            var hidden = _weigthsIH * input + _biasIH;
            hidden.Sigmoid();

            var output = _weightsHO * hidden + _biasHO;
            output.Sigmoid();

            var expected = new Matrix(target);
            var outputError =  expected - output;
            var dOutput = Matrix.Dsigmoid(output);
            var hiddenT = Matrix.Transpose(hidden);

            var gradient = Matrix.Hadamard(dOutput, outputError);
            gradient = gradient * _learningRate;

            _biasHO += gradient;

            var weigthsHODeltas = gradient * hiddenT;
            _weightsHO += weigthsHODeltas;

            var weightsHOT = Matrix.Transpose(_weightsHO);
            var hiddenError = weightsHOT * outputError;
            var dHidden = Matrix.Dsigmoid(hidden);
            var inputT = Matrix.Transpose(input);

            var gradientH = Matrix.Hadamard(dHidden, hiddenError);
            gradientH = gradientH * _learningRate;

            _biasIH += gradientH;
            var weightsIHDeltar = gradientH * inputT;
            _weigthsIH += weightsIHDeltar;
        }

        public float[] Predict(float[] arr)
        {
            var input = new Matrix(arr);

            var hidden = _weigthsIH * input + _biasIH;
            hidden.Sigmoid();
            
            var output = _weightsHO * hidden + _biasHO;
            output.Sigmoid();
            
            var result = output.ToArray();
            return result;
        }
    }
}