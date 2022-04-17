#include "dataset.hh"
#include <filesystem>
#include <iostream>
#include <fstream>

ChessDataSet::ChessDataSet(std::string path) {
	this->createDataset(path);
}

void ChessDataSet::createDataset(std::string path){
	/* std::vector<ChessDataTest> data_temp;
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
	} */
	std::cout << "Created dataset" << std::endl;
}

void ChessDataSet::read(std::string filename, torch::Tensor data) {
	torch::load(data, filename);
}

void ChessDataSet::write(std::string filename, torch::Tensor data) {
	torch::save(data, filename);
}

torch::data::Example<> ChessDataSet::get(size_t index) {
	return this->data[index];
}

torch::optional<size_t> ChessDataSet::size() const {
	return this->data.size();
}

