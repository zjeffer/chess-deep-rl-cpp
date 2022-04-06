#include "dataset.hh"
#include <filesystem>
#include <iostream>
#include <fstream>

ChessDataSet::ChessDataSet(std::string path) {
	this->createDataset(path);
}

void ChessDataSet::createDataset(std::string path){
	for (auto& p : std::filesystem::directory_iterator(path)) {
		std::string file_name = p.path().string();
		std::cout << "Reading file: " << file_name << std::endl;
		this->addDataFromFile(file_name);
	}
}

void ChessDataSet::addDataFromFile(std::string file_name) {
	std::cout << "Adding data from file: " << file_name << std::endl;
	struct ChessData chessData;
	
	std::ifstream file_stream(file_name);
	while(file_stream.read((char*)&chessData, sizeof(struct ChessData))) {
		printf("%d %d %f\n", chessData.input.size(), chessData.policy.size(), chessData.value);
		std::cout << "Pushing to data" << std::endl;
		std::cout << this->data.size() << std::endl;

		this->data.push_back(chessData);
		printf("Pushed to data\n");
	}
	file_stream.close();
}

void ChessDataSet::write(std::string filename, std::vector<ChessData> data) {
	try {
		std::filesystem::create_directory("memory");
	} catch (std::filesystem::filesystem_error& e) {
		std::cerr << e.what() << std::endl;
	}

	std::cout << "Writing to memory/" << filename << std::endl;
	FILE *file = fopen(("memory/" + filename).c_str(), "w");
	if (file == NULL) {
		perror("Error opening file when writing ChessDataSet");
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < data.size(); i++) {
		fwrite(&data[i], sizeof(struct ChessData), 1, file);
	}
	fclose(file);
}

torch::data::Example<> ChessDataSet::get(size_t index) {
	struct ChessData chessData = data[index];
	return {
		torch::from_blob(chessData.input.data(), {1, 19, 8, 8}), 
		torch::cat({
			torch::from_blob(chessData.policy.data(), {1, 73, 8, 8}), 
			torch::tensor(chessData.value)
		}, 1)};
}

torch::optional<size_t> ChessDataSet::size() const {
	return this->data.size();
}

