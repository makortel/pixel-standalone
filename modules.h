#ifndef modules_h
#define modules_h

namespace gpuClustering {
  constexpr uint32_t MaxNumModules = 2000;
  constexpr uint16_t InvId = 9999;  // must be > MaxNumModules
}  // namespace gpuClustering

inline
int countModules(const uint16_t *id, int size) {
  int modules = 0;
  for (int i = 0; i < size; ++i) {
    if (id[i] == gpuClustering::InvId)
      continue;
    auto j = i - 1;
    while (j >= 0 and id[j] == gpuClustering::InvId) {
      --j;
    }
    if (j < 0 or id[j] != id[i]) {
      ++modules;
    }
  }
  return modules;
}

#endif  // modules_h
