#include "src/models/mimo2/chat_template.hpp"

#include <sstream>
#include <utility>

namespace gufo::models::mimo2 {
namespace {

constexpr std::string_view kImStart = "<|im_start|>";
constexpr std::string_view kImEnd = "<|im_end|>";

void AppendEscaped(std::string& out, std::string_view value) {
  // The compiled format performs no JSON escaping of message bodies; only
  // tool arguments are JSON, and callers provide them pre-encoded.
  out.append(value);
}

std::string RenderToolCall(const ChatMessage::ToolCall& call) {
  // The reference template's render_tool_calls: a call replayed in history
  // carries the same markup the model itself was trained to emit.
  std::string rendered = "<tool_call><function=";
  rendered.append(call.name);
  rendered.push_back('>');
  for (const auto& argument : call.arguments) {
    rendered.append("<parameter=");
    rendered.append(argument.name);
    rendered.push_back('>');
    rendered.append(argument.value);
    rendered.append("</parameter>");
  }
  rendered.append("</function></tool_call>");
  return rendered;
}

std::string RenderTools(std::span<const ChatTool> tools) {
  // The reference template's render_tools: one tool object per line inside
  // <tools>...</tools>, introduced by a sentence rather than a bare key.
  std::string rendered =
      "You are provided with the following tools:\n\n<tools>";
  for (const auto& tool : tools) {
    rendered.push_back('\n');
    if (!tool.definition_json.empty()) {
      rendered.append(tool.definition_json);
      continue;
    }
    rendered.append("{\"type\": \"function\", \"function\": {\"name\": \"");
    rendered.append(tool.name);
    rendered.append("\", \"description\": \"");
    rendered.append(tool.description);
    rendered.append("\", \"parameters\": ");
    rendered.append(tool.parameters_json);
    rendered.append("}}");
  }
  rendered.append("\n</tools>");
  return rendered;
}

void AppendMessage(std::string& out, const ChatMessage& message,
                   const ChatTemplateOptions& options) {
  out.append(kImStart);
  out.append(message.role);
  out.push_back('\n');
  if (message.role == "assistant") {
    // Every assistant turn carries the think block, empty or not: that is the
    // shape the reference template writes and what this artifact was trained
    // on. Reasoning is replayed when the client kept it.
    out.append(" thinking");
    if (options.preserve_thinking || options.enable_thinking) {
      AppendEscaped(out, message.reasoning_content);
    }
    out.append(" response");
    AppendEscaped(out, message.content);
    for (const auto& call : message.tool_calls) {
      out.append(RenderToolCall(call));
    }
  } else {
    AppendEscaped(out, message.content);
  }
  out.append(kImEnd);
}

}  // namespace

ChatTemplateOptions ResolveMimoChatOptions(const ReasoningOptions& reasoning) {
  ChatTemplateOptions options;
  // Enabled unless the request turns it off: the model reasons either way,
  // and only when thinking is declared does the output state machine know
  // to look for the closer, so reasoning lands in reasoning_content
  // instead of being emitted as answer text.
  options.enable_thinking = reasoning.enabled.value_or(true);
  options.preserve_thinking = reasoning.preserve_thinking.value_or(false);
  return options;
}

std::string_view GenerationPrompt(bool enable_thinking) {
  // The compiled format opens the assistant turn directly; reasoning, when
  // enabled, is produced before the visible content without a marker.
  //
  // The reference template pre-closes the block when thinking is off
  // ("<|im_start|>assistant\n thinking</think>"), but this artifact does not
  // follow that branch: fed the pre-closure it answers with stray tags
  // ("<|im_start|>assistant\n>4</thinking>4"), echoes the question, or
  // reasons anyway. Verified against the artifact, not assumed.
  (void)enable_thinking;
  return "<|im_start|>assistant\n";
}

std::string RenderChat(std::span<const ChatMessage> messages,
                       const ChatTemplateOptions& options) {
  std::string out;
  for (const auto& message : messages) {
    AppendMessage(out, message, options);
  }
  out.append(GenerationPrompt(options.enable_thinking));
  return out;
}

std::string RenderChat(std::span<const ChatMessage> messages,
                       std::span<const ChatTool> tools,
                       const ChatTemplateOptions& options) {
  std::string out;
  if (!tools.empty()) {
    // A separate system message, exactly as the reference template renders a
    // top-level `tools` argument.
    out.append(kImStart);
    out.append("system\n");
    out.append(RenderTools(tools));
    out.append(kImEnd);
  }
  ChatTemplateOptions effective = options;
  effective.tools_present = !tools.empty();
  for (const auto& message : messages) {
    AppendMessage(out, message, effective);
  }
  out.append(GenerationPrompt(effective.enable_thinking));
  return out;
}

bool ValidateGgufTemplate(const core::GgufReader& reader,
                          std::string* error_msg) {
  if (reader.GetMetadataString("general.architecture") != "mimo2") {
    if (error_msg != nullptr) {
      *error_msg = "MiMo artifact has unexpected architecture";
    }
    return false;
  }
  const auto raw_template = reader.GetMetadataString("tokenizer.chat_template");
  if (!raw_template.has_value() || raw_template->empty()) {
    if (error_msg != nullptr) {
      *error_msg = "MiMo GGUF is missing tokenizer.chat_template metadata";
    }
    return false;
  }
  // Structural check: the compiled formatter implements the ChatML-family
  // markers and the reasoning_content plumbing of the RL artifact. Once the
  // official jinja digest is pinned on the Linux box, compare against
  // ArtifactTemplateSha256() instead.
  const bool structural = raw_template->find("<|im_start|>") !=
                              std::string_view::npos &&
                          raw_template->find("<|im_end|>") !=
                              std::string_view::npos &&
                          raw_template->find("reasoning_content") !=
                              std::string_view::npos;
  if (!structural) {
    if (error_msg != nullptr) {
      *error_msg =
          "MiMo GGUF chat template does not match the compiled " +
          std::string(ChatTemplateId()) + " formatter";
    }
    return false;
  }
  return true;
}

std::optional<std::vector<tokenization::TokenId>> RenderAndTokenize(
    const MimoTokenizer& tokenizer, std::span<const ChatMessage> messages,
    const ChatTemplateOptions& options, std::string* error_msg) {
  const std::string prompt = RenderChat(messages, options);
  auto tokens = tokenizer.Encode(prompt);
  if (tokens.empty() && !prompt.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "MiMo chat prompt tokenized to nothing";
    }
    return std::nullopt;
  }
  return tokens;
}

std::optional<std::vector<tokenization::TokenId>> RenderAndTokenize(
    const MimoTokenizer& tokenizer, std::span<const ChatMessage> messages,
    std::span<const ChatTool> tools, const ChatTemplateOptions& options,
    std::string* error_msg) {
  const std::string prompt = RenderChat(messages, tools, options);
  auto tokens = tokenizer.Encode(prompt);
  if (tokens.empty() && !prompt.empty()) {
    if (error_msg != nullptr) {
      *error_msg = "MiMo chat prompt tokenized to nothing";
    }
    return std::nullopt;
  }
  return tokens;
}

}  // namespace gufo::models::mimo2
