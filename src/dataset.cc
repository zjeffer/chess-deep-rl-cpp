#include "dataset.hh"
#include <filesystem>
#include <fstream>
#include <iostream>

ChessDataSet::ChessDataSet(std::string path) {
    int draws = 0, whiteWins = 0, blackWins = 0;
    // for every folder in path/
    for (const auto &game : std::filesystem::directory_iterator(path)) {
        if (!g_Running) return;
        // if folder is a folder and starts with game-
        if (std::filesystem::is_directory(game.path()) && game.path().string().find(path + "/game-") == 0) {
            int result = -2;
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
                } catch (const std::exception& e) {
                    LOG(WARNING) << "Could not load tensor from file: " << move.path() << "\n" << e.what();
                    exit(EXIT_FAILURE);
                }

                if (input.sizes() != torch::IntArrayRef({119, 8, 8}) || outputs.sizes() != torch::IntArrayRef({4673})) {
                    LOG(WARNING) << "Invalid sizes for file " << move.path() << ": " << input.sizes() << " " << outputs.sizes();
                    exit(EXIT_FAILURE);
                }
                result = outputs.slice(0, 4672, 4673).view({1}).item<int>();
                this->data.push_back(std::make_pair(input, outputs));
            }
            if (result == 0) {
                draws++;
            } else if (result == 1) {
                whiteWins++;
            } else if (result == -1) {
                blackWins++;
            } else {
                LOG(WARNING) << "Invalid result for game " << game.path();
                exit(EXIT_FAILURE);
            }
        } else {
            LOG(DEBUG) << "Skipping " << game.path().string();
        }
    }
    LOG(INFO) << "Created dataset of size " << this->data.size();
    LOG(INFO) << "Dataset contains " << draws << " draws, " << whiteWins << " white wins, and " << blackWins << " black wins.";
}

torch::data::Example<> ChessDataSet::get(size_t index) {
    torch::Tensor input = this->data[index].first;
    torch::Tensor output = this->data[index].second;
    return {input, output};
}

torch::optional<size_t> ChessDataSet::size() const {
    return this->data.size();
}
