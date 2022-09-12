#define CL_USE_DEPRECATED_OPENCL_2_0_APIS	// using OpenCL 1.2, some functions deprecated in OpenCL 2.0
#define __CL_ENABLE_EXCEPTIONS				// enable OpenCL exemptions

// C++ standard library and STL headers
#include <iostream>
#include <vector>
#include <fstream>
#include <list>

// OpenCL header, depending on OS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "common.h"

// function to save the message into a file
void saveMessageToFile(std::vector<cl_char>* v, std::string filename)
{
	std::ofstream file(filename);
	for (int i = 0; i < v->size(); i++)
		file << v->at(i);
	file.close();
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
	std::vector<cl_char> charVec(2754), charVecEncryptOutput(2754), charVecDecryptOutput(2754), encryptionLookUp(255), decryptionLookUp(255);
	cl::Buffer encryptInputBuffer, encryptOutputBuffer, decryptInputBuffer, decryptOutputBuffer, lookupMapBuffer;

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
		if (!build_program(&program, &context, "task2c.cl"))
		{
			// if OpenCL program build error
			quit_program("OpenCL program build error.");
		}

		// initialize lookup tables
		// load the file containing the character mapping
		char ch;
		std::fstream encoderFile("encoder.txt", std::fstream::in);
		int i = 0;
		std::vector<char> temp(26);
		while (encoderFile >> std::noskipws >> ch)
		{
			temp[i] = ch;
			i++;
		}
		std::cout << std::endl;
		encoderFile.close();
		// store mapping in vector
		int alphabetNum = 65;
		for (int a = 0; a < temp.size(); a++)
		{
			encryptionLookUp[alphabetNum] = temp[a];
			encryptionLookUp[alphabetNum + 32] = temp[a];
			decryptionLookUp[(int)temp[a]] = alphabetNum;
			alphabetNum++;
		}

		// read plaintest.txt and store each character in a list
		std::fstream fin("plaintext.txt", std::fstream::in);
		i = 0;
		while (fin >> std::noskipws >> ch)
		{
			charVec[i] = ch;
			i++;
		}
		fin.close();

		// create a kernel
		kernel = cl::Kernel(program, "task2c");

		// create command queue
		queue = cl::CommandQueue(context, device);

		// create buffers
		encryptInputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_char) * charVec.size(), &charVec[0]);
		lookupMapBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_char) * encryptionLookUp.capacity(), &encryptionLookUp[0]);
		encryptOutputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_char) * charVecEncryptOutput.capacity());

		// set kernel arguments
		kernel.setArg(0, encryptInputBuffer);
		kernel.setArg(1, lookupMapBuffer);
		kernel.setArg(2, encryptOutputBuffer);

		// enqueue kernel for execution
		cl::NDRange offset(0);
		cl::NDRange globalSize(charVec.size()/2);

		queue.enqueueNDRangeKernel(kernel, offset, globalSize);

		std::cout << "Encryption kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;

		// enqueue command to read from device to host memory
		queue.enqueueReadBuffer(encryptOutputBuffer, CL_TRUE, 0, sizeof(cl_char) * charVecEncryptOutput.capacity(), &charVecEncryptOutput[0]);

		// decryption
		lookupMapBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_char) * decryptionLookUp.capacity(), &decryptionLookUp[0]);
		decryptInputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_char) * charVecEncryptOutput.capacity(), &charVecEncryptOutput[0]);
		decryptOutputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_char) * charVecDecryptOutput.capacity());
		kernel.setArg(0, decryptInputBuffer);
		kernel.setArg(1, lookupMapBuffer);
		kernel.setArg(2, decryptOutputBuffer);
		queue.enqueueNDRangeKernel(kernel, offset, globalSize);
		queue.enqueueReadBuffer(decryptOutputBuffer, CL_TRUE, 0, sizeof(cl_char) * charVecDecryptOutput.capacity(), &charVecDecryptOutput[0]);

		std::cout << "Decryption kernel enqueued." << std::endl;
		std::cout << "--------------------" << std::endl;
		
		// check the results
		cl_bool isSame = CL_TRUE;
		for (int i = 0; i < charVecDecryptOutput.size(); i++)
		{
			if (toupper(charVec[i]) != charVecDecryptOutput[i])
			{
				std::cout << "Text does not match!" << std::endl;
				std::cout << toupper(charVec[i]) << " vs " << charVecDecryptOutput[i] << std::endl;
				isSame = CL_FALSE;
				break;
			}
		}
		if (isSame)
		{
			std::cout << "Text matches" << std::endl;
		}
		std::cout << "--------------------" << std::endl;

		// Save output to a file
		saveMessageToFile(&charVecEncryptOutput, "ciphertext2c.txt");
		saveMessageToFile(&charVecDecryptOutput, "decrypted2c.txt");
		std::cout << "Messages saved to file." << std::endl;
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