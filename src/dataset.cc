#include "dataset.hh"
#include <filesystem>
#include <fstream>
#include <iostream>

ChessDataSet::ChessDataSet(std::string path) {
    // for every folder in path/
    for (const auto &game : std::filesystem::directory_iterator(path)) {
        // if folder is a folder and starts with game-
        if (std::filesystem::is_directory(game.path()) && game.path().string().find(path + "/game-") == 0) {
            // for every move in the folder:
            for (const auto &move : std::filesystem::directory_iterator(game.path())) {
                if (!std::filesystem::is_regular_file(move.path()) || move.path().string().find("input") == std::string::npos) {
                    // if not a file or not an input tensor, skip
                    continue;
                }
                std::string output_path;
                torch::Tensor input;
                torch::Tensor outputs;
                try {
                    torch::load(input, move.path());
                    // also load corresponding output: output path is the same as input path, but with "output" instead of "input"
                    output_path = move.path().string().replace(move.path().string().find("input"), 5, "output");
                    torch::load(outputs, output_path.c_str());
                } catch (std::exception &e) {
                    std::cerr << "Could not load tensor from file: " << move.path() << std::endl;
                    std::cerr << e.what() << std::endl;
                    exit(1);
                }

                if (input.sizes() != torch::IntArrayRef({119, 8, 8}) || outputs.sizes() != torch::IntArrayRef({4673})) {
                    std::cerr << "Invalid sizes for file " << move.path() << ": " << input.sizes() << " " << outputs.sizes() << std::endl;
                    exit(1);
                }
                
                this->data.push_back(std::make_pair(input, outputs));
            }
        } else {
            std::cout << "Skipping " << game.path().string() << std::endl;
        }
    }
    std::cout << "Created dataset of size " << this->data.size() << std::endl;
}

torch::data::Example<> ChessDataSet::get(size_t index) {
    torch::Tensor input = this->data[index].first;
    torch::Tensor output = this->data[index].second;
    return {input, output};
}

torch::optional<size_t> ChessDataSet::size() const {
    return this->data.size();
}
