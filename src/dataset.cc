#include "dataset.hh"
#include <filesystem>
#include <iostream>
#include <fstream>

ChessDataSet::ChessDataSet(std::string path) {
	this->createDataset(path);
}

void ChessDataSet::createDataset(std::string path){
	std::vector<ChessDataTest> data_temp;
	for (auto& p : std::filesystem::directory_iterator(path)) {
		std::string filename = p.path().string();
		std::cout << "Checking file " << filename << " ... ";
		if (filename.find("game-") != std::string::npos) {
			this->read(filename, &data_temp);
			std::cout << "Added data from" << std::endl;
			std::cout << "Size so far: " << data_temp.size() << std::endl;
		} else {
			std::cout << "Skipping..." << std::endl;
		}
	}
	for(int i = 0; i < (int)data_temp.size(); i++){
		ChessData cd;
		cd.input = data_temp[i].input;
		cd.policy = data_temp[i].policy;
		cd.value = data_temp[i].value;
		this->data.push_back(cd);
	}
	std::cout << "Created dataset" << std::endl;
}

void ChessDataSet::read(std::string filename, std::vector<ChessDataTest>* data_temp) {
	std::cout << "Adding data from file: " << filename << std::endl;
	struct ChessDataTest chessData;
	
	std::ifstream file_stream(filename);
	while(file_stream.read((char*)&chessData, sizeof(struct ChessData))) {

		struct ChessDataTest new_data;
		new_data.input = chessData.input;
		new_data.policy = chessData.policy;
		new_data.value = chessData.value;

		data_temp->push_back(new_data);
	}
	file_stream.close();
	std::cout << "Closed file" << std::endl;
}

void ChessDataSet::write(std::string filename, std::vector<ChessDataTest> data) {
	// write to memory/ folder
	try {
		std::filesystem::create_directory("memory");
	} catch (std::filesystem::filesystem_error& e) {
		std::cerr << "Error creating directory: " << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}

	try {
		std::ofstream file_stream("memory/" + filename, std::ios::out | std::ios::binary);
		for (auto& d : data) {
			file_stream.write((char*)&d, sizeof(struct ChessDataTest));
		}
		file_stream.close();
	} catch (std::filesystem::filesystem_error& e) {
		std::cerr << "Error writing to file: " << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}
}

torch::data::Example<> ChessDataSet::get(size_t index) {
	return this->data[index];
}

torch::optional<size_t> ChessDataSet::size() const {
	return this->data.size();
}

