using System;
using System.Text;

namespace NeuralNetwork
{
    public class Matrix
    {
        private readonly float [,] _matrix;

        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            _matrix = new float[rows, columns];
        }

        public Matrix(float[] array)
        {
            Rows = array.Length;
            Columns = 1;
            _matrix = new float[array.Length, 1];

            for (int i = 0; i < array.Length; i++)
                _matrix[i, 0] = array[i];
        }

        public int Rows { get; }
        public int Columns { get; }

        public float[] ToArray()
        {
            var array = new float[Rows * Columns];

            for (int i = 0, pos = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++, pos++)
                    array[pos] = _matrix[i, j];

            return array;
        }

        public void Map(Func<float, float> callback)
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++)
                    _matrix[i, j] = callback(_matrix[i, j]);
        }

        public void Print()
        {
            var sb = new StringBuilder();

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                    sb.AppendFormat("{0} ", _matrix[i, j]);

                sb.AppendLine();
            }

            Console.WriteLine(sb.ToString());
        }

        public void Randomize()
        {
            var r = new Random();
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++)
                    _matrix[i, j] = r.Next() * (1f / int.MaxValue);
        }

        public void Sigmoid() => Map(x => 1f / (1 + MathF.Exp(-x)));

        public static Matrix Map(Matrix input, Func<float, float> callback)
        {
            var matrix = new Matrix(input.Rows, input.Columns);

            for (int i = 0; i < input.Rows; i++)
                for (int j = 0; j < input.Columns; j++)
                    matrix._matrix[i, j] = callback(input._matrix[i, j]);

            return matrix;
        }

        public static Matrix Dsigmoid(Matrix input) => Map(input, x => x * (1 - x));

        public static Matrix Transpose(Matrix input)
        {
            var matrix = new Matrix(input.Columns, input.Rows);

            for (int i = 0; i < input.Rows; i++)
                for (int j = 0; j < input.Columns; j++)
                    matrix._matrix[j, i] = input._matrix[i, j];

            return matrix;
        }

        public static Matrix Hadamard(Matrix a, Matrix b)
        {
            var matrix = new Matrix(a.Rows, a.Columns);

            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Columns; j++)
                    matrix._matrix[i, j] = a._matrix[i, j] * b._matrix[i, j];

            return matrix;
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            var matrix = new Matrix(a.Rows, a.Columns);

            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Columns; j++)
                    matrix._matrix[i, j] = a._matrix[i, j] + b._matrix[i, j];

            return matrix;
        }

        public static Matrix operator -(Matrix a, Matrix b)
        {
            var matrix = new Matrix(a.Rows, a.Columns);

            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Columns; j++)
                    matrix._matrix[i, j] = a._matrix[i, j] - b._matrix[i, j];

            return matrix;
        }

        public static Matrix operator *(Matrix input, float scalar) => Map(input, x => x * scalar);

        public static Matrix operator *(Matrix a, Matrix b)
        {
            var matrix = new Matrix(a.Rows, b.Columns);

            for (int i = 0; i < matrix.Columns; i++)
            {
                for (int j = 0; j < matrix.Rows; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < a.Columns; k++)
                    {
                        var elm1 = a._matrix[j,k];
                        var elm2 = b._matrix[k,i];
                        sum += elm1 * elm2;
                    }
                    matrix._matrix[j, i] = sum;
                }
            }

            return matrix;
        }
    }
}