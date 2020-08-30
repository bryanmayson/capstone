package edu.monash.fit.eduard.grid.operator.lic;
import java.io.IOException;


public class OpenCLConnection {
	// Allows data to be passed to the OpenCL enviroment and back
	public native float[] computeOpenCL(float[] bufferdata,int nrow,int ncol,double[] weights,double[] weightsSummed,float halfslope,int iterations);
	
	public native boolean testPassing (float[] bufferdata,int nrow,int ncol,double[] weights,double[] weightsSummed,float halfslope,int iterations);
	
	//this is just for testing and will check if the dll connects correctly - it should never be able to run in normal usage
	public static void main(String[] args) {
		System.loadLibrary("OpenCLConnection");
		OpenCLConnection OpenCL = new OpenCLConnection();
		float[] test_data = {10.10f};
		double[] a = {10d};
		boolean check = OpenCL.testPassing(test_data, 1, 1, a, a, 1f, 1);
		
		System.out.println("Data passing check: " + check);
				
	}
	//handles passing data into the OpenCL code
	public float[] compute(float[] bufferdata,int nrow,int ncol,double[] weights,double[] weightsSummed,float halfslope,int iterations) throws IOException,RuntimeException {
		System.loadLibrary("OpenCLConnection");
		OpenCLConnection OpenCL = new OpenCLConnection();
		float[] result;
		System.out.println("loaded");
		Boolean testDll=false;
		//function should return true is data is passed correctly into the dll
		testDll=OpenCL.testPassing(bufferdata, iterations, iterations, weightsSummed, weightsSummed, halfslope, iterations);
		
		if(testDll) {
			System.out.println("dll function correctly called");
		}
		else
		{
			System.out.println("Failed to pass data into dll");
			throw new IOException("Failed to pass data into dll");
		}

		float[] testbuff=new float[100];
		for(int i=0;i<100;i++)
		{
			testbuff[i]=0;
		}
		result = OpenCL.computeOpenCL(bufferdata, nrow, ncol, weights, weightsSummed, halfslope,iterations);
		/*for(int i=0;i<10;i++)
		{
			for(int e=0;e<10;e++)
			{
				System.out.print(":");
				System.out.print(result[e+10*i]);
			}
			System.out.println(":");
		}*/
		//dll will return a error number indicating the location of the error, as well as the error code for that error
		if(result.length==2)
		{
			//an error has occured
			throw new RuntimeException(String.format("An error has occured with OpenCL. dll error code %d OpenCL error code %d",(int) result[0],(int) result[1]));
		}
			
		return result;
	}
}
