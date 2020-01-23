using System;
using System;
using Numpy;
using NumSharp;      
using Keras;
using Keras.Models;
using Keras.Layers;
/// <summary>
///  Train 
/// </summary>
namespace ConsoleApp1
{
    class Program1
    {
        static void Main(string[] args)
        {
              
            //Mock data fix the file address
            NDarray dataset = Numpy.np.loadtxt(fname: "C:\\Natan\\csharp\\trial11\\pima-indians-diabetes.csv", delimiter: ",");
            var X = dataset[":,0: 8"];
            var Y = dataset[":, 8"];
            var model = new Sequential();
            model.Add(new Dense(12, input_dim: 8, kernel_initializer: "uniform", activation: "relu"));
            model.Add(new Dense(8, kernel_initializer: "uniform", activation: "relu"));
            model.Add(new Dense(1, activation: "sigmoid"));
            model.Compile(optimizer: "adam", loss: "binary_crossentropy", metrics: new string[] { "accuracy" });
            model.Fit(X, Y, batch_size: 10, epochs: 150, verbose: 1);
            model.Save("modelAA.h5");
            double[] scores = model.Evaluate(X, Y);
            foreach (double sc in scores)
            {
                Console.WriteLine(sc);
                //Console.WriteLine(scmodel..metrics_names[1], scores[1] * 100))
            }

            Console.WriteLine("Hello World! we learned");
            Console.ReadKey();

        }
    }
}


 
