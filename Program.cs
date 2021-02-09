using System;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            var network = new Network(2, 3, 1);

            var inputs = new float[][]
            {
                new float[] { 1, 1 },
                new float[] { 1, 0 },
                new float[] { 0, 1 },
                new float[] { 0, 0 },
            };

            var outputs = new float[][]
            {
                new float[] { 0 },
                new float[] { 1 },
                new float[] { 1 },
                new float[] { 0 }
            };

            var train = true;
            var counter = 1;
            while (train)
            {
                for (int i = 0; i < 10000; i++)
                {
                    var index = new Random().Next(0, inputs.Length);
                    network.Train(inputs[index], outputs[index]);
                }

                if (network.Predict(new float[] {0, 0})[0] < 0.04 && network.Predict(new float[] {1, 0})[0] > 0.98) 
                {
                    train = false;
                    Console.WriteLine($"terminou com {counter} iterações");
                }
                counter++;
            }
            
        }
    }
}
