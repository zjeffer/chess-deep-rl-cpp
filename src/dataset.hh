#pragma once


#undef slots
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/nn.h>
#include <torch/script.h>
#define slots Q_SLOTS

#include "types.hh"
#include "common.hh"

using ChessData = std::pair<torch::Tensor, torch::Tensor>;

class ChessDataSet : public torch::data::datasets::Dataset<ChessDataSet> {
  public:
	explicit ChessDataSet(std::string path);
	
	void read(std::string filename, torch::Tensor data);

	static void write(std::string filename, torch::Tensor data);

	torch::data::Example<> get(size_t index) override;

	torch::optional<size_t> size() const override;

  private:
	std::vector<ChessData> data;

};
