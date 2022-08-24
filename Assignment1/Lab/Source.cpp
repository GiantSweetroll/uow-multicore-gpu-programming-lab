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

using namespace std;

int main(void)
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device
	string outputString;			// string for output
	int outputInt;					// int for output	

	try {
		// select an OpenCL device
		if (!select_one_device(&platform, &device))
		{
			// if no device selected
			quit_program("Device not selected.");
		}

		// create a context from device
		context = cl::Context(device);

		// create command queue
		queue = cl::CommandQueue(context, device);

		// build the program
		if (!build_program(&program, &context, device, "source.cl"))
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// get and output the platform name
		platform.getInfo(CL_PLATFORM_NAME, &outputString);
		cout << "Platform Name: " << outputString << std::endl;

		// get and output device type
		cl_device_type type;
		device.getInfo(CL_DEVICE_TYPE, &type);
		if (type == CL_DEVICE_TYPE_CPU)
			cout << "Type: " << "CPU" << endl;
		else if (type == CL_DEVICE_TYPE_GPU)
			cout << "Type: " << "GPU" << endl;
		else
			cout << "Type: " << "Other" << endl;

		// get and output device name
		outputString = device.getInfo<CL_DEVICE_NAME>();
		cout << "Name: " << outputString << endl;

		// get and output the amount of compute units
		outputInt = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
		cout << "Number of compute units: " << outputInt << endl;

		// get and output the maximum work group size
		outputInt = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		cout << "Maximum work group size: " << outputInt << endl;

		// get and output put the maximum work item sizes
		vector<size_t> maxWorkItemSizes;
		maxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
		cout << "--------------------" << endl;
		cout << "Max work item sizes:" << endl;
		for (int i = 0; i < maxWorkItemSizes.size(); i++) 
		{
			cout << maxWorkItemSizes[i] << endl;
		}
		cout << "--------------------" << endl;

		// get and output the global memory size
		cl_ulong globalMemSize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
		cout << "Global memory size: " << globalMemSize << " bytes" << endl;
		cout << "--------------------" << endl;

		// check if device supports the cl_khr_fp16 extension
		string supportedExtensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
		if (supportedExtensions.find("cl_khr_fp16") != string::npos) // npos is what string.find() returns if no match was found
		{
			cout << "This device supports the cl_khr_fp16 extension!" << endl;
		}
		else
		{
			cout << "This device does not support the cl_khr_fp16 extension." << endl;
		}
		cout << "--------------------" << endl;

		// find and display the number of kernels in the program
		vector<cl::Kernel> allKernels;

		// create all kernels in the program
		program.createKernels(&allKernels);

		// output the number of kernels
		cout << "The number of kernels: " << allKernels.size() << endl;

		// output kernel names
		for (int i = 0; i < allKernels.size(); i++)
		{
			outputString = allKernels[i].getInfo<CL_KERNEL_FUNCTION_NAME>();
			cout << "Kernel " << i << ": " << outputString << endl;
		}
		cout << "--------------------" << endl;
	}
	// catch any OpenCL function errors
	catch (cl::Error e) {
		// call function to handle errors
		handle_error(e);
	}

#ifdef _WIN32
	// wait for a keypress on Windows OS before exiting
	cout << "\npress a key to quit...";
	cin.ignore();
#endif

	return 0;
}
