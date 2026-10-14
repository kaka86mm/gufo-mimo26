#ifndef GUFO_SERVER_TOOL_SERVING_HPP_
#define GUFO_SERVER_TOOL_SERVING_HPP_

#include <cstdlib>

namespace gufo::server {

/// Some deployments serve tool calls free-form: the chat template lists the
/// tools, the model writes the <tool_call> markup itself, and the response
/// parser extracts it. GUFO_TOOLS_FREEFORM=1 skips the grammar, and with it
/// the per-token mask over the whole vocabulary (152k pieces here), which is
/// expensive enough to starve the TP2 peer's exchange timeout. The grammar
/// path stays the default.
[[nodiscard]] inline bool ToolsServedFreeform() {
  return std::getenv("GUFO_TOOLS_FREEFORM") != nullptr;
}

}  // namespace gufo::server

#endif  // GUFO_SERVER_TOOL_SERVING_HPP_