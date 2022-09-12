// C++ standard library and STL headers
#include <iostream>
#include <vector>
#include <fstream>
#include <list>

using namespace std;

// function to encrypt/decrypt the message
void encrypt(list<char>* ls, int n)
{
	list<char>::iterator it;
	for (it = ls->begin(); it != ls->end(); ++it)
	{
		// encrypt char
		char c = *it;
		c = toupper(c);		// make it uppercase

		// only shift alphabetical characters
		if (c >= 65 && c <= 90)
			c += n;
		
		// put it back into the list
		*it = c;
	}
}

// function to save the message into a file
void saveMessageToFile(list<char>* ls, string filename)
{
	ofstream encryptedFile(filename);
	list<char>::iterator it;
	for (it = ls->begin(); it != ls->end(); ++it)
		encryptedFile << *it;
	encryptedFile.close();
}

int main(void)
{
	// read plaintest.txt and store each character in a list
	char ch;
	list<char> charList;
	fstream fin("plaintext.txt", fstream::in);

	while (fin >> noskipws >> ch)
	{
		charList.push_back(ch);
	}
	fin.close();

	// select n value
	int n;
	cout << "Select value of n: ";
	cin >> n;
	cout << endl;

	// encrypt characters
	encrypt(&charList, n);

	// store encrypted message into a file
	saveMessageToFile(&charList, "ciphertext2a.txt");

	// decrypt message
	encrypt(&charList, n * -1);

	// store decrypted message into a file
	saveMessageToFile(&charList, "decrypted2a.txt");
	
	return 0;
}
