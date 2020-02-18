#ifndef analyzer_cupla_h_
#define analyzer_cupla_h_

class Input;
class Output;

namespace cupla_seq_seq_sync {
  void analyze(Input const& input, Output& output, double& totaltime);
}

namespace cupla_seq_seq_async {
  void analyze(Input const& input, Output& output, double& totaltime);
}

namespace cupla_tbb_seq_async {
  void analyze(Input const& input, Output& output, double& totaltime);
}

namespace cupla_omp2_seq_async {
  void analyze(Input const& input, Output& output, double& totaltime);
}

namespace cupla_omp4_omp4_async {
  void analyze(Input const& input, Output& output, double& totaltime);
}

namespace cupla_cuda_async {
  void analyze(Input const& input, Output& output, double& totaltime);
}

#endif  // analyzer_cupla_h_
