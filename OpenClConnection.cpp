#include "edu_monash_fit_eduard_grid_operator_lic_OpenCLConnection.h"
#include <string.h>
#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <direct.h>

#define FILE_ERROR -1
#define PROGRAM_ERROR -2
#define BUILD_ERROR -3
#define DATA_ERROR -4
#define QUEUE_ERROR -5

//location of the kernal source file relative to the program working directory
const char* sourcePath = "lic.cl";

//Function requires a specific name and headers to communicate properly with java
//this function is given the data from java and will perform the lic algorithm useing OpenCl
JNIEXPORT jfloatArray JNICALL Java_edu_monash_fit_eduard_grid_operator_lic_OpenCLConnection_computeOpenCL
(JNIEnv* env, jobject obj, jfloatArray bufferdata, jint nrow, jint ncol, jdoubleArray weights, jdoubleArray weightsSummed, jfloat halfslope, jint iterations)
{
	//convert java arrays to arrays usable by C
	jfloat* nativeData = env->GetFloatArrayElements(bufferdata, NULL);
	jdouble* nativeWeights = env->GetDoubleArrayElements(weights, NULL);
	jdouble* nativeSummedWeights = env->GetDoubleArrayElements(weightsSummed, NULL);
	int weightLength = env->GetArrayLength(weights);
	int ndLength= env->GetArrayLength(bufferdata);

	//return value on a fail state
	jfloatArray failArray = env->NewFloatArray(2);


	/*printf("weight length: %d\n", weightLength);
	printf("nd length: %d\n", ndLength);
	int j;
	for (j = 0; j < weightLength; j++)
	{
		printf("%lf:", nativeWeights[j]);
	}*/

	//load the kernal source file into a string
	FILE* sFile;
	char buff[200];
	_getcwd(buff, 200);
	printf("%s\n", buff);
	int ferr = fopen_s(&sFile, sourcePath, "rb");
	if (ferr != 0)
	{
		printf("file error: %d", ferr);
		float failValues[2] = {FILE_ERROR,ferr};
		env->SetFloatArrayRegion(failArray, 0, 2, failValues);
		return failArray;
	}
	fseek(sFile, 0, SEEK_END);
	int fileLen = ftell(sFile);
	fseek(sFile, 0, SEEK_SET);
	char* source = (char*)malloc(fileLen + 1);
	fread(source, 1, fileLen, sFile);
	fclose(sFile);
	source[fileLen] = 0;


	//Get a platform.
	cl_platform_id platform;
	clGetPlatformIDs(1, &platform, NULL);

	//Find a gpu device.
	cl_device_id device;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	//log some device info
	char devName[50];
	clGetDeviceInfo(device, CL_DEVICE_VERSION, 50, devName, NULL);
	printf(devName);
	printf("\n");

	//Create a context and command queue on that device.
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, NULL);

	//Perform runtime source compilation, and obtain kernel entry point.
	cl_int err;
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
	if (err != 0)
	{
		printf("failed to create program with error: %d", err);
		float failValues[2] = { PROGRAM_ERROR,err };
		env->SetFloatArrayRegion(failArray, 0, 2, failValues);
		return failArray;
	}

	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

	size_t log_size;
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

	//Log build results for debugging purposes
	char* log = (char*)malloc(log_size);
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
	printf("%s\n", log);
	free(log);
	printf("=============================================\n");
	if (err != 0)
	{
		printf("failed to build program with error: %d", err);
		float failValues[2] = { BUILD_ERROR,err };
		env->SetFloatArrayRegion(failArray, 0, 2, failValues);
		return failArray;
	}

	//create kernal
	cl_kernel kernel = clCreateKernel(program, "lic", NULL);

	//convert arrays into buffers that the OpenCL kernal can use
	//old opencl doesn't have good support for 2D buffers (image objects) so it's easier to use a 1D buffer and pretend it's 2D
	cl_mem hieghtMap = clCreateBuffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, ndLength* sizeof(cl_float), (void*)nativeData, &err);
	if (err != 0)
	{
		printf("failed to initalise heightmap with error: %d", err);
		float failValues[2] = { DATA_ERROR,err };
		env->SetFloatArrayRegion(failArray, 0, 2, failValues);
		return failArray;
	}
	cl_mem GaussianWeights = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, weightLength*sizeof(cl_double), (void*)nativeWeights, &err);
	if (err != 0)
	{
		printf("failed to initalise gaussian weights with error: %d", err);
		float failValues[2] = { DATA_ERROR,err };
		env->SetFloatArrayRegion(failArray, 0, 2, failValues);
		return failArray;
	}
	cl_mem GaussianWeightsSummed = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weightLength*sizeof(cl_double), (void*)nativeSummedWeights, &err);
	if (err != 0)
	{
		printf("failed to initalise summed gaussian weights with error: %d", err);
		float failValues[2] = { DATA_ERROR,err };
		env->SetFloatArrayRegion(failArray, 0, 2, failValues);
		return failArray;
	}

	printf("hmapsize in bytes: %d\n", nrow * ncol * sizeof(float));

	//Assign the input arguments of the kernal
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&hieghtMap);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&GaussianWeights);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&GaussianWeightsSummed);
	clSetKernelArg(kernel, 3, sizeof(cl_int), &weightLength);

	printf("start\n");

	//indicates the size and dimentions of work item IDs
	//ID is 2D because we are working with 2d coordinates, even though the data buffer is a 1D array
	size_t global_work_size[2] = { ncol, nrow };

	int i;
	//itterate lic kernal
	//add the kernal for one iteration to the queue, wait until it's finished, then repeat
	for (i = 0; i < iterations; i++)
	{
		cl_int queueError=clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, 0, 0, NULL, NULL);
		clFinish(queue);
		if (queueError != 0)
		{
			printf("failed to initalise heightmap with error: %d", queueError);
			float failValues[2] = { QUEUE_ERROR,queueError };
			env->SetFloatArrayRegion(failArray, 0, 2, failValues);
			return failArray;
		}
	}
	clFinish(queue);

	//retrieve results from OpenCL buffer
	cl_float* ptr;
	ptr = (cl_float*)clEnqueueMapBuffer(queue, hieghtMap, CL_TRUE, CL_MAP_READ, 0, nrow*ncol * sizeof(cl_float), 0, NULL, NULL, &err);
	//cl_int queueError = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, 0, 0, NULL, NULL);
	//clFinish(queue);
	if (err != 0)
	{
		printf("failed to retrive data with error: %d", err);
		float failValues[2] = { DATA_ERROR,err };
		env->SetFloatArrayRegion(failArray, 0, 2, failValues);
		return failArray;
	}
	printf("done\n");


	free(source);
	//pur results into java array and return
	jfloatArray retFloatArr = env->NewFloatArray(ndLength);
	env->SetFloatArrayRegion(retFloatArr, 0, nrow * ncol, ptr);
	return retFloatArr;

}

//testing the passing of data between programs
JNIEXPORT jboolean JNICALL Java_edu_monash_fit_eduard_grid_operator_lic_OpenCLConnection_testPassing
(JNIEnv*, jobject obj, jfloatArray bufferdata, jint nrow, jint ncol, jdoubleArray weights, jdoubleArray weightsSummed, jfloat halfslope, jint iterations) {
	bool check = true;
	return check;
}