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
	std::vector<cl_int> data(2);
	cl::Buffer dataBuffer;
	std::string devExt;				// string for device extensions

	try {
		// select an OpenCL device
		if (!select_one_device(&platform, &device))
		{
			// if no device selected
			quit_program("Device not selected.");
		}

		// check whether device supports int32 base atomics for local memory
		devExt = device.getInfo<CL_DEVICE_EXTENSIONS>();

		std::string ext("cl_khr_local_int32_base_atomics");
		size_t found = devExt.find(ext);

		if (found != std::string::npos) 
		{
			std::cout << "local memory int32 base atomics supported" << std::endl;
		}
		else
		{
			std::cout << "local memory int32 base atomics NOT supported" << std::endl;
		}

		// create a context from device
		context = cl::Context(device);

		// build the program
		if(!build_program(&program, &context, "atomic.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "atomic");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// create buffers
		dataBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * 2);

		// set kernel arguments
		kernel.setArg(0, dataBuffer);

		cl::NDRange offset(0);
		cl::NDRange globalSize(8);
		cl::NDRange localSize(4);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(dataBuffer, CL_TRUE, 0, sizeof(cl_int) * 2, &data[0]);

		// output results
		std::cout << "Increment: " << data[0] << std::endl;
		std::cout << "Atomic increment: " << data[1] << std::endl;
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
