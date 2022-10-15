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

// callback functions
void CL_CALLBACK kernel_complete(cl_event e, cl_int status, void* data) 
{
	// display the data
	printf("%s", (char*)data);
}

void CL_CALLBACK check_data(cl_event e, cl_int status, void* data) 
{
	cl_bool check;
	cl_float *bufferData;

	// check the data whether the contents are all 5.0
	bufferData = (cl_float*)data;
	check = CL_TRUE;
	for (int i = 0; i < 4096; i++) {
		if (bufferData[i] != 5.0) {
			check = CL_FALSE;
			break;
		}
	}
	if (check)
		std::cout << "The data is accurate." << std::endl;
	else
		std::cout << "The data is not accurate." << std::endl;
}

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

	// declare data and memory objects
	std::vector<cl_float> data(4096);
	cl::Buffer resultBuffer;

	// declare events
	cl::Event kernelEvent, readEvent;

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
		if(!build_program(&program, &context, "callback.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "callback");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// create buffers
		resultBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * 4096);

		// set kernel arguments
		kernel.setArg(0, resultBuffer);

		// enqueue kernel for execution
		queue.enqueueTask(kernel, NULL, &kernelEvent);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(resultBuffer, CL_FALSE, 0, sizeof(cl_float) * 4096, &data[0], NULL, &readEvent);

		// set event callback functions
		const char* kernelMsg = "The kernel finished successfully.\n\0";
		kernelEvent.setCallback(CL_COMPLETE, &kernel_complete, (void*)kernelMsg);
		readEvent.setCallback(CL_COMPLETE, &check_data, (void*)&data[0]);

		// wait for enqueued commands to complete
		queue.finish();
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
