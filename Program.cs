using System;
using Numpy;
using Keras;
using Keras.Models;
using Microsoft.ML.OnnxRuntime;

// Test
namespace ConsoleApp2
{
    class Program
    {
        static void Main(string[] args)
        {
            /*SessionOptions options = new SessionOptions();
            /*options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;*/

            Console.WriteLine("qqqqwwwwwwwwwyay");
            var session = new InferenceSession("C:\\Users\\natank\\Desktop\\model_2good.onnx");
            Console.WriteLine("wwwwwyay");

            //Mock data fix the file address
            NDarray xtr = np.load("C:\\Natan\\fff\\fff\\NKLA_D_100a2200.npy");
            NDarray xtr2 = np.load("C:\\Natan\\fff\\fff\\NKLA_D_100a2318.npy");
            Console.WriteLine(xtr2.shape);

            var model = Model.LoadModel("C:\\Natan\\fff\\fff\\_reg_best.h5");
            Console.WriteLine("yay");

            // Fix the model address
            NDarray scores = model.Predict(xtr2);
            Console.WriteLine(scores.shape);
            Console.WriteLine(scores[0,0]);

            scores = model.Predict(xtr);
            Console.WriteLine(scores.shape);

            /*     foreach (double sc in scores)
            {
                Console.WriteLine(sc);
                //Console.WriteLine(scmodel..metrics_names[1], scores[1] * 100))
            }*/


            Console.WriteLine("Hello World! we tested");
            Console.ReadKey();
        }
    }
}
