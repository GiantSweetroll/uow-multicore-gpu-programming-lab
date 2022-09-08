#define CL_USE_DEPRECATED_OPENCL_2_0_APIS	// using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS				// enable OpenCL exemptions

// C++ standard library and STL headers
#include <iostream>
#include <vector>
#include <fstream>

// OpenCL header, depending on OS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "common.h"

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	std::vector<cl_float> mat(16), vec(4), result(4), correctResult(4);
	cl::Buffer matBuffer, vecBuffer, resultBuffer;

	// initialise values
	for (int i = 0; i < 16; i++) 
	{
		mat[i] = i * 2.0f;
	}
	for (int i = 0; i < 4; i++) 
	{
		vec[i] = i * 3.0f;
		correctResult[0] += mat[i] * vec[i];
		correctResult[1] += mat[i + 4] * vec[i];
		correctResult[2] += mat[i + 8] * vec[i];
		correctResult[3] += mat[i + 12] * vec[i];
	}

	try {
		// select an OpenCL device
		if (!select_one_device(&platform, &device))
		{
			// if no device selected
			quit_program("Device not selected.");
		}

		// create a context from device
		context = cl::Context(device);

		// build the program
		if(!build_program(&program, &context, "matvec.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "matvec_mult");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// create buffers
		matBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * 16, &mat[0]);
		vecBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * 4, &vec[0]);
		resultBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * 4);

		// set kernel arguments
		kernel.setArg(0, matBuffer);
		kernel.setArg(1, vecBuffer);
		kernel.setArg(2, resultBuffer);

		// enqueue kernel for execution
		//queue.enqueueTask(kernel);

		cl::NDRange offset(0);
		cl::NDRange globalSize(4);	// 4 work-units per kernel

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);
			   
		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, sizeof(cl_float) * 4, &result[0]);

		// check the results
		if ((result[0] == correctResult[0]) && (result[1] == correctResult[1]) && 
			(result[2] == correctResult[2]) && (result[3] == correctResult[3])) 
		{
			std::cout << "Matrix-vector multiplication successful." << std::endl;

			printf("| %4.1f %4.1f %4.1f %4.1f |   |%4.1f |    |  %4.1f |\n", mat[0], mat[1], mat[2], mat[3], vec[0], result[0]);
			printf("| %4.1f %4.1f %4.1f %4.1f | * |%4.1f |  = | %4.1f |\n", mat[4], mat[5], mat[6], mat[7], vec[1], result[1]);
			printf("| %4.1f %4.1f %4.1f %4.1f |   |%4.1f |    | %4.1f |\n", mat[8], mat[9], mat[10], mat[11], vec[2], result[2]);
			printf("| %4.1f %4.1f %4.1f %4.1f |   |%4.1f |    | %4.1f |\n", mat[12], mat[13], mat[14], mat[15], vec[3], result[3]);
		}
		else 
		{
			std::cout << "Matrix-vector multiplication unsuccessful." << std::endl;
		}
	}
	// catch any OpenCL function errors
	catch (cl::Error e) {
		// call function to handle errors
		handle_error(e);
	}

#ifdef _WIN32
	// wait for a keypress on Windows OS before exiting
	std::cout << "\npress a key to quit...";
	std::cin.ignore();
#endif

	return 0;
}
