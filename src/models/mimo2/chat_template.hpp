#ifndef GUFO_MODELS_MIMO2_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_MIMO2_CHAT_TEMPLATE_HPP_

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/reasoning.hpp"
#include "src/models/mimo2/tokenizer.hpp"

namespace gufo::models::mimo2 {

struct ChatMessage {
  struct ToolArgument {
    std::string name;
    std::string value;
    bool is_string{true};
  };

  struct ToolCall {
    std::string name;
    std::vector<ToolArgument> arguments;
    std::string id;
  };

  std::string role;
  std::string content;
  std::string reasoning_content;
  std::vector<ToolCall> tool_calls;
  std::string tool_call_id;
};

struct ChatTool {
  std::string name;
  std::string description;
  std::string parameters_json{"{}"};
  /// Complete function definition, including optional fields such as strict.
  std::string definition_json;
};

struct ChatTemplateOptions {
  // Literal encoder options; user-facing defaults are resolved below.
  bool enable_thinking{false};
  bool preserve_thinking{false};
  bool tools_present{false};
  bool require_tool_call{false};
};

/// User-facing default follows the RL artifact's chat format: thinking
/// enabled when the request enables it or leaves it on auto.
[[nodiscard]] ChatTemplateOptions ResolveMimoChatOptions(
    const ReasoningOptions& reasoning);

/// The generation prompt that continues an assistant turn.
[[nodiscard]] std::string_view GenerationPrompt(bool enable_thinking);

/// Renders the compiled MiMo-V2 conversation format (ChatML-family
/// delimiters, assistant reasoning before the content, JSON tool calls).
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     const ChatTemplateOptions& options = {});
[[nodiscard]] std::string RenderChat(std::span<const ChatMessage> messages,
                                     std::span<const ChatTool> tools,
                                     const ChatTemplateOptions& options = {});

/// Rejects MiMo artifacts whose embedded template is not the version the
/// compiled formatter implements. The check is structural (the markers the
/// renderer emits must appear in the jinja) until the official template's
/// SHA-256 is pinned from a verified artifact on the Linux box.
[[nodiscard]] bool ValidateGgufTemplate(const core::GgufReader& reader,
                                        std::string* error_msg = nullptr);

[[nodiscard]] constexpr std::string_view ChatTemplateId() noexcept {
  return "mimo-v2-flash-rl-compiled-v1";
}

/// Empty until the artifact template digest is pinned (see
/// ValidateGgufTemplate); callers must not treat empty as a match.
[[nodiscard]] constexpr std::string_view ArtifactTemplateSha256() noexcept {
  return "";
}

/// Formats messages and tokenizes the rendered prompt with the model's
/// tokenizer; same call shape as the Qwen serving glue so the serve layer
/// can be wired by duplication.
[[nodiscard]] std::optional<std::vector<tokenization::TokenId>>
RenderAndTokenize(const MimoTokenizer& tokenizer,
                  std::span<const ChatMessage> messages,
                  const ChatTemplateOptions& options = {},
                  std::string* error_msg = nullptr);

[[nodiscard]] std::optional<std::vector<tokenization::TokenId>>
RenderAndTokenize(const MimoTokenizer& tokenizer,
                  std::span<const ChatMessage> messages,
                  std::span<const ChatTool> tools,
                  const ChatTemplateOptions& options = {},
                  std::string* error_msg = nullptr);

}  // namespace gufo::models::mimo2

#endif  // GUFO_MODELS_MIMO2_CHAT_TEMPLATE_HPP_
