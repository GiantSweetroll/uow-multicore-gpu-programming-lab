#define CL_USE_DEPRECATED_OPENCL_2_0_APIS	// using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS				// enable OpenCL exemptions

// C++ standard library and STL headers
#include <iostream>
#include <vector>
#include <sstream>

// OpenCL header, depending on OS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

// functions to handle errors
#include "error.h"

// to avoid having to use prefixes
//using namespace std;
//using namespace cl;

// function that allows the user to select a device
// it displays the available platform and device options
// returns whether selection was successful, the selected device and its platform
bool select_one_device(cl::Platform* platfm, cl::Device* dev)
{
	std::vector<cl::Platform> platforms;	// available platforms
	std::vector< std::vector<cl::Device> > platformDevices;	// devices available for each platform
	std::string outputString;				// string for output
	unsigned int i, j;								// counters

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

		// check whether valid option was selected
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

int main(void)
{
	cl::Platform platform;		// device's platform
	cl::Device device;			// device used
	cl::Context context;		// context for the device
	std::string outputString;	// string for output
	unsigned int i;				// counter

	try {
		// select an OpenCL device
		if (!select_one_device(&platform, &device))
		{
			// if no device selected
			quit_program("Device not selected.");
		}

		// create a context from device
		context = cl::Context(device);

		// check devices in the context
		std::cout << "\nDevices in the context:" << std::endl;

		// get devices in the context
		std::vector<cl::Device> contextDevices = context.getInfo<CL_CONTEXT_DEVICES>();

		// output names of devices in the context
		for (i = 0; i < contextDevices.size(); i++)
		{
			outputString = contextDevices[i].getInfo<CL_DEVICE_NAME>();
			std::cout << "  Device " << i << ": " << outputString << std::endl;
		}
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