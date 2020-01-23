using System;
using Numpy;
using NumSharp;
using Keras;
using Keras.Models;
using Keras.Layers;

namespace synth0
{
    class Program2
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            //NDarray xtr = Numpy.np.load("C:\\Users\\natank\\Desktop\\codesynth\\myfile.npy");
            NDarray xtr = Numpy.np.load("C:\\Natan\\fff\\fff\\NKLA_D_100a2200.npy");

            var model = Model.LoadModel("C:\\Users\\natank\\Desktop\\codesynth\\bbb.h5" );
            Console.WriteLine("yay");
            NDarray scores = model.Predict(xtr);
            

        }
    }
}
