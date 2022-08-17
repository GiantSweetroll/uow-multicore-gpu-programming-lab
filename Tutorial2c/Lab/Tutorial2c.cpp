#define CL_USE_DEPRECATED_OPENCL_2_0_APIS	// using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS				// enable OpenCL exemptions

// C++ standard library and STL headers
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

// OpenCL header, depending on OS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

// functions to handle errors
#include "error.h"

// function prototypes
bool select_one_device(cl::Platform* platfm, cl::Device* dev);
bool build_program(cl::Program* prog, const cl::Context* ctx, const std::string filename);

// builds program from given filename
bool build_program(cl::Program* prog, const cl::Context* ctx, const std::string filename)
{
	// get devices from the context
	std::vector<cl::Device> contextDevices = ctx->getInfo<CL_CONTEXT_DEVICES>();

	// open input file stream to .cl file
	std::ifstream programFile(filename);

	// check whether file was opened
	if (!programFile.is_open())
	{
		std::cout << "File not found." << std::endl;
		return false;
	}

	// create program string and load contents from the file
	std::string programString(std::istreambuf_iterator<char>(programFile), (std::istreambuf_iterator<char>()));

	// create program source from one input string
	cl::Program::Sources source(1, std::make_pair(programString.c_str(), programString.length() + 1));
	// create program from source
	*prog = cl::Program(*ctx, source);

	// try to build program
	try {
		// build the program for the devices in the context
		prog->build(contextDevices);

		std::cout << "Program build: Successful" << std::endl;
		std::cout << "--------------------" << std::endl;
	}
	catch (cl::Error e) {
		// if failed to build program
		if (e.err() == CL_BUILD_PROGRAM_FAILURE)
		{
			// output program build log
			std::cout << e.what() << ": Failed to build program." << std::endl;

			// check build status for all all devices in context
			for (unsigned int i = 0; i < contextDevices.size(); i++)
			{
				// get device's program build status and check for error
				// if build error, output build log
				if (prog->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(contextDevices[i]) == CL_BUILD_ERROR)
				{
					// get device name and build log
					std::string outputString = contextDevices[i].getInfo<CL_DEVICE_NAME>();
					std::string build_log = prog->getBuildInfo<CL_PROGRAM_BUILD_LOG>(contextDevices[i]);

					std::cout << "Device - " << outputString << ", build log:" << std::endl;
					std::cout << build_log << "--------------------" << std::endl;
				}
			}

			return false;
		}
		else
		{
			// call function to handle errors
			handle_error(e);
		}
	}

	return true;
}

int main(void) 
{
	cl::Platform platform;			// device's platform
	cl::Device device;				// device used
	cl::Context context;			// context for the device
	cl::Program program;			// OpenCL program object
	cl::Kernel kernel;				// a single kernel object
	cl::CommandQueue queue;			// commandqueue for a context and device

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
		if(!build_program(&program, &context, "blank.cl")) 
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// create a kernel
		kernel = cl::Kernel(program, "blank");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// set kernel argument
		int a = 0;
		kernel.setArg(0, a);

		// enqueue kernel for execution
		queue.enqueueTask(kernel);

		std::cout << "Kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;
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

// allows the user to select a device, displays the available platform and device options
// returns whether selection was successful, the selected device and its platform
bool select_one_device(cl::Platform* platfm, cl::Device* dev)
{
	std::vector<cl::Platform> platforms;	// available platforms
	std::vector< std::vector<cl::Device> > platformDevices;	// devices available for each platform
	std::string outputString;				// string for output
	unsigned int i, j;						// counters

	try {
		// get the number of available OpenCL platforms
		cl::Platform::get(&platforms);
		std::cout << "Number of OpenCL platforms: " << platforms.size() << std::endl;

		// find and store the devices available to each platform
		for (i = 0; i < platforms.size(); i++)
		{
			std::vector<cl::Device> devices;		// available devices

			// get all devices available to the platform
			platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

			// store available devices for the platform
			platformDevices.push_back(devices);
		}

		// display available platforms and devices
		std::cout << "--------------------" << std::endl;
		std::cout << "Available options:" << std::endl;

		// store options as platform and device indices
		std::vector< std::pair<int, int> > options;
		unsigned int optionCounter = 0;	// option counter

		// for all platforms
		for (i = 0; i < platforms.size(); i++)
		{
			// for all devices per platform
			for (j = 0; j < platformDevices[i].size(); j++)
			{
				// display options
				std::cout << "Option " << optionCounter << ": Platform - ";

				// platform vendor name
				outputString = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
				std::cout << outputString << ", Device - ";

				// device name
				outputString = platformDevices[i][j].getInfo<CL_DEVICE_NAME>();
				std::cout << outputString << std::endl;

				// store option
				options.push_back(std::make_pair(i, j));
				optionCounter++; // increment option counter
			}
		}

		std::cout << "\n--------------------" << std::endl;
		std::cout << "Select a device: ";

		std::string inputString;
		unsigned int selectedOption;	// option that was selected

		std::getline(std::cin, inputString);
		std::istringstream stringStream(inputString);

		// check whether valid option selected
		// check if input was an integer
		if (stringStream >> selectedOption)
		{
			char c;

			// check if there was anything after the integer
			if (!(stringStream >> c))
			{
				// check if valid option range
				if (selectedOption >= 0 && selectedOption < optionCounter)
				{
					// return the platform and device
					int platformNumber = options[selectedOption].first;
					int deviceNumber = options[selectedOption].second;

					*platfm = platforms[platformNumber];
					*dev = platformDevices[platformNumber][deviceNumber];

					return true;
				}
			}
		}
		// if invalid option selected
		std::cout << "\n--------------------" << std::endl;
		std::cout << "Invalid option." << std::endl;
	}
	// catch any OpenCL function errors
	catch (cl::Error e) {
		// call function to handle errors
		handle_error(e);
	}

	return false;
}

