package edu.monash.fit.eduard.grid.operator.lic;
import java.io.IOException;
import java.util.Objects;
import edu.monash.fit.eduard.grid.Grid;


public final class LICOpenCLOperator extends LineIntegralConvolutionOperator {
	
	/***
	 *  Subclass of LIC Operator
	 *  Specifically made to pass raw grid data into the OpenCL application and retrieve filter application back
	 * 	
	 * 	Before returning the results, the raw data of produced by OpenCL would be converted back to its Grid format
	 * 
	 * */
	
	private int iterations ;
	
	private OpenCLConnection OpenCL = new OpenCLConnection();
	
	public LICOpenCLOperator(int iterations,float halfLineLength,float sharpening,float sharpeningLimit) {
		super( halfLineLength, sharpening, sharpeningLimit);
		this.iterations = iterations;
	}
	
	public Grid operate(Grid src) {

			logStart();
			Objects.requireNonNull(src, getName() + ": source grid is null");
	
			if (!src.isWellFormed()) {
			    throw new IllegalStateException(getName() + ": grid is not well formed");
			}
			
			
			float[] data = src.getBufferData();
			int nrows= src.getRows();
			int ncols= src.getCols();

		float[] result = new float[0];
		try {
			result = OpenCL.compute(data,nrows,ncols,gaussianWeights,gaussianWeightsSummed,halfBoxFilterLineLength,iterations);
		} catch (IOException e) {
			System.out.println(e.getMessage());
			System.exit(-1);
		}
		finally {
            logEnd();
        }
		src.updateBufferData(result);
		return src;


	}
}
