#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace tarakanov_d_linear_stretching_omp {

class TaskParallel : public ppc::core::Task {
 public:
  explicit TaskParallel(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<unsigned char> inputImage_;
  std::vector<unsigned char> outputImage_;

  int rc_size_{};
};

}  // namespace tarakanov_d_linear_stretching_omp
