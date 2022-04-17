#ifndef DATASET_HH
#define DATASET_HH

#include <torch/torch.h>
#include <torch/data.h>

#include "types.hh"


class ChessDataSet : public torch::data::datasets::Dataset<ChessDataSet> {
  public:
	ChessDataSet(std::string path);
	
	void createDataset(std::string path);

	void read(std::string filename, std::vector<ChessDataTest>* data);

	static void write(std::string filename, std::vector<ChessDataTest> data);

	torch::data::Example<> get(size_t index) override;

	torch::optional<size_t> size() const override;

  private:
	std::vector<ChessData> data;

};

#endif // DATASET_HH