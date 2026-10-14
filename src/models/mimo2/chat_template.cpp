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
  std::string arguments = "{";
  for (std::size_t i = 0; i < call.arguments.size(); ++i) {
    const auto& argument = call.arguments[i];
    if (i != 0) {
      arguments.append(", ");
    }
    arguments.append("\"");
    arguments.append(argument.name);
    arguments.append("\": ");
    arguments.append(argument.is_string ? "\"" : "");
    arguments.append(argument.value);
    arguments.append(argument.is_string ? "\"" : "");
  }
  arguments.append("}");
  std::ostringstream rendered;
  rendered << "{\"name\": \"" << call.name << "\", \"arguments\": "
           << arguments << ", \"id\": \"" << call.id << "\"}";
  return rendered.str();
}

std::string RenderTools(std::span<const ChatTool> tools) {
  std::ostringstream rendered;
  rendered << "[";
  for (std::size_t i = 0; i < tools.size(); ++i) {
    const auto& tool = tools[i];
    if (i != 0) {
      rendered << ", ";
    }
    if (!tool.definition_json.empty()) {
      rendered << tool.definition_json;
      continue;
    }
    rendered << "{\"type\": \"function\", \"function\": {\"name\": \""
             << tool.name << "\", \"description\": \"" << tool.description
             << "\", \"parameters\": " << tool.parameters_json << "}}";
  }
  rendered << "]";
  return rendered.str();
}

void AppendMessage(std::string& out, const ChatMessage& message,
                   const ChatTemplateOptions& options) {
  out.append(kImStart);
  out.append(message.role);
  out.push_back('\n');
  if (message.role == "assistant" && !message.reasoning_content.empty() &&
      (options.preserve_thinking || options.enable_thinking)) {
    AppendEscaped(out, message.reasoning_content);
    out.append(kImEnd);
    out.push_back('\n');
  }
  if (!message.tool_calls.empty()) {
    for (const auto& call : message.tool_calls) {
      out.append("<tool_call>");
      out.append(RenderToolCall(call));
      out.append("</tool_call>");
      out.push_back('\n');
    }
  } else {
    AppendEscaped(out, message.content);
    if (!message.content.empty() || message.role != "tool") {
      out.push_back('\n');
    }
  }
  out.append(kImEnd);
  out.push_back('\n');
}

}  // namespace

ChatTemplateOptions ResolveMimoChatOptions(const ReasoningOptions& reasoning) {
  ChatTemplateOptions options;
  options.enable_thinking = reasoning.enabled.value_or(false);
  options.preserve_thinking = reasoning.preserve_thinking.value_or(false);
  return options;
}

std::string_view GenerationPrompt(bool enable_thinking) {
  // The compiled format opens the assistant turn directly; reasoning, when
  // enabled, is produced before the visible content without a marker.
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
    out.append(kImStart);
    out.append("system\ntools: ");
    out.append(RenderTools(tools));
    out.push_back('\n');
    out.append(kImEnd);
    out.push_back('\n');
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
