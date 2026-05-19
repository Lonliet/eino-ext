/*
 * Copyright 2026 CloudWeGo Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package agenticclaude

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/cloudwego/eino/schema"
	claudeschema "github.com/cloudwego/eino/schema/claude"
)

func TestToolSearchResultToBlockParam(t *testing.T) {
	blockParam, err := toolSearchResultToBlockParam(&schema.ToolSearchFunctionToolResult{
		CallID: "call_1",
		Result: &schema.ToolSearchResult{
			Tools: []*schema.ToolInfo{
				{Name: "tool_a"},
				{Name: "tool_b"},
			},
		},
	})
	if err != nil {
		t.Fatalf("toolSearchResultToBlockParam() error = %v", err)
	}

	got := mustJSON(t, blockParam)
	for _, want := range []string{
		`"type":"tool_result"`,
		`"tool_use_id":"call_1"`,
		`"tool_name":"tool_a","type":"tool_reference"`,
		`"tool_name":"tool_b","type":"tool_reference"`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("toolSearchResultToBlockParam() json = %s, want substring %s", got, want)
		}
	}
}

func TestServerToolResultToBlockParam(t *testing.T) {
	t.Run("tool search search result", func(t *testing.T) {
		blockParam, err := serverToolResultToBlockParam(schema.NewContentBlock(&schema.ServerToolResult{
			CallID: "call_1",
			Name:   string(ServerToolNameToolSearchToolBm25),
			Content: &ServerToolResult{
				ToolSearchToolBm25: &ToolSearchToolResult{
					Type: ToolSearchToolResultTypeSearchResult,
					SearchResult: &ToolSearchToolSearchResult{
						ToolReferences: []*ToolSearchToolReference{
							{ToolName: "tool_a"},
						},
					},
				},
			},
		}))
		if err != nil {
			t.Fatalf("serverToolResultToBlockParam() error = %v", err)
		}

		got := mustJSON(t, blockParam)
		for _, want := range []string{
			`"type":"tool_search_tool_result"`,
			`"tool_use_id":"call_1"`,
			`"type":"tool_search_tool_search_result"`,
			`"tool_name":"tool_a"`,
		} {
			if !strings.Contains(got, want) {
				t.Fatalf("serverToolResultToBlockParam() json = %s, want substring %s", got, want)
			}
		}
	})

	t.Run("tool search error", func(t *testing.T) {
		blockParam, err := serverToolResultToBlockParam(schema.NewContentBlock(&schema.ServerToolResult{
			CallID: "call_2",
			Name:   string(ServerToolNameToolSearchToolRegex),
			Content: &ServerToolResult{
				ToolSearchToolRegex: &ToolSearchToolResult{
					Type:  ToolSearchToolResultTypeError,
					Error: &ToolSearchToolResultError{Code: "invalid_query"},
				},
			},
		}))
		if err != nil {
			t.Fatalf("serverToolResultToBlockParam() error = %v", err)
		}
		got := mustJSON(t, blockParam)
		for _, want := range []string{
			`"type":"tool_search_tool_result"`,
			`"type":"tool_search_tool_result_error"`,
			`"error_code":"invalid_query"`,
		} {
			if !strings.Contains(got, want) {
				t.Fatalf("serverToolResultToBlockParam() json = %s, want substring %s", got, want)
			}
		}
	})
}

func TestServerToolUseToContentBlock(t *testing.T) {
	t.Run("tool search bm25 input", func(t *testing.T) {
		block, err := serverToolUseToContentBlock(anthropic.ServerToolUseBlock{
			ID:    "call_1",
			Name:  anthropic.ServerToolUseBlockNameToolSearchToolBm25,
			Input: map[string]any{"query": "find tools"},
		})
		if err != nil {
			t.Fatalf("serverToolUseToContentBlock() error = %v", err)
		}
		if block.ServerToolCall == nil {
			t.Fatalf("serverToolUseToContentBlock() returned nil server tool call")
		}
		args, ok := block.ServerToolCall.Arguments.(*ServerToolCallArguments)
		if !ok {
			t.Fatalf("server tool call arguments type = %T", block.ServerToolCall.Arguments)
		}
		if args.ToolSearchToolBm25 == nil || args.ToolSearchToolBm25.Query != "find tools" {
			t.Fatalf("tool search bm25 args = %#v", args.ToolSearchToolBm25)
		}
	})

	t.Run("web fetch nil input keeps empty args", func(t *testing.T) {
		block, err := serverToolUseToContentBlock(anthropic.ServerToolUseBlock{
			ID:   "call_2",
			Name: anthropic.ServerToolUseBlockNameWebFetch,
		})
		if err != nil {
			t.Fatalf("serverToolUseToContentBlock() error = %v", err)
		}
		args, ok := block.ServerToolCall.Arguments.(*ServerToolCallArguments)
		if !ok {
			t.Fatalf("server tool call arguments type = %T", block.ServerToolCall.Arguments)
		}
		if args.WebFetch == nil || args.WebFetch.URL != "" {
			t.Fatalf("web fetch args = %#v", args.WebFetch)
		}
	})
}

func TestToDeltaResponseMeta(t *testing.T) {
	meta := toDeltaResponseMeta(anthropic.MessageDeltaEvent{
		Usage: anthropic.MessageDeltaUsage{
			InputTokens:              10,
			CacheReadInputTokens:     3,
			CacheCreationInputTokens: 2,
			OutputTokens:             7,
		},
		Delta: anthropic.MessageDeltaEventDelta{
			StopReason:   anthropic.StopReasonEndTurn,
			StopSequence: "done",
			StopDetails: anthropic.RefusalStopDetails{
				Category:    anthropic.RefusalStopDetailsCategoryBio,
				Explanation: "blocked",
			},
		},
	})

	if meta == nil || meta.TokenUsage == nil || meta.ClaudeExtension == nil {
		t.Fatalf("toDeltaResponseMeta() = %#v", meta)
	}
	if meta.TokenUsage.PromptTokens != 15 || meta.TokenUsage.CompletionTokens != 7 || meta.TokenUsage.TotalTokens != 22 {
		t.Fatalf("token usage = %#v", meta.TokenUsage)
	}
	if meta.TokenUsage.PromptTokenDetails.CachedTokens != 3 {
		t.Fatalf("cached tokens = %d, want 3", meta.TokenUsage.PromptTokenDetails.CachedTokens)
	}
	if meta.ClaudeExtension.StopReason != string(anthropic.StopReasonEndTurn) {
		t.Fatalf("stop reason = %q", meta.ClaudeExtension.StopReason)
	}
	if meta.ClaudeExtension.StopSequence != "done" {
		t.Fatalf("stop sequence = %q", meta.ClaudeExtension.StopSequence)
	}
	if meta.ClaudeExtension.StopDetails == nil || meta.ClaudeExtension.StopDetails.Explanation != "blocked" {
		t.Fatalf("stop details = %#v", meta.ClaudeExtension.StopDetails)
	}
}

func TestToAnthropicTextCitations(t *testing.T) {
	citations, err := toAnthropicTextCitations([]*claudeschema.TextCitation{
		{
			Type: claudeschema.TextCitationTypeWebSearchResultLocation,
			WebSearchResultLocation: &claudeschema.CitationWebSearchResultLocation{
				CitedText:      "snippet",
				Title:          "result title",
				URL:            "https://example.com",
				EncryptedIndex: "enc_1",
			},
		},
	})
	if err != nil {
		t.Fatalf("toAnthropicTextCitations() error = %v", err)
	}
	if len(citations) != 1 || citations[0].OfWebSearchResultLocation == nil {
		t.Fatalf("citations = %#v", citations)
	}
	got := citations[0].OfWebSearchResultLocation
	if got.URL != "https://example.com" || got.EncryptedIndex != "enc_1" {
		t.Fatalf("web search result citation = %#v", got)
	}
}

func TestToWebFetchDocumentBlockParam(t *testing.T) {
	t.Run("invalid mime type", func(t *testing.T) {
		_, err := toWebFetchDocumentBlockParam(&WebFetchDocument{
			Source: &WebFetchDocumentSource{
				MIMEType: "application/json",
				Data:     "{}",
			},
		})
		if err == nil || !strings.Contains(err.Error(), `invalid web fetch content mime type "application/json"`) {
			t.Fatalf("toWebFetchDocumentBlockParam() error = %v", err)
		}
	})

	t.Run("plain text document", func(t *testing.T) {
		blockParam, err := toWebFetchDocumentBlockParam(&WebFetchDocument{
			Title: "doc",
			Citations: &WebFetchDocumentCitations{
				Enabled: true,
			},
			Source: &WebFetchDocumentSource{
				MIMEType: "text/plain",
				Data:     "hello",
			},
		})
		if err != nil {
			t.Fatalf("toWebFetchDocumentBlockParam() error = %v", err)
		}
		got := mustJSON(t, blockParam)
		for _, want := range []string{
			`"title":"doc"`,
			`"type":"text"`,
			`"data":"hello"`,
			`"enabled":true`,
		} {
			if !strings.Contains(got, want) {
				t.Fatalf("toWebFetchDocumentBlockParam() json = %s, want substring %s", got, want)
			}
		}
	})
}

func TestToAnthropicMessages(t *testing.T) {
	input := []*schema.AgenticMessage{
		schema.SystemAgenticMessage("follow system"),
		{
			Role: schema.AgenticRoleTypeUser,
			ContentBlocks: []*schema.ContentBlock{
				schema.NewContentBlock(&schema.UserInputText{Text: "hello"}),
				schema.NewContentBlock(&schema.UserInputImage{URL: "https://example.com/image.png"}),
				schema.NewContentBlock(&schema.UserInputFile{URL: "https://example.com/doc.pdf"}),
				schema.NewContentBlock(&schema.FunctionToolResult{
					CallID: "call_fn",
					Name:   "get_weather",
					Content: []*schema.FunctionToolResultContentBlock{
						{
							Type: schema.FunctionToolResultContentBlockTypeText,
							Text: &schema.UserInputText{Text: "sunny"},
						},
					},
				}),
				schema.NewContentBlock(&schema.ToolSearchFunctionToolResult{
					CallID: "call_search",
					Result: &schema.ToolSearchResult{
						Tools: []*schema.ToolInfo{{Name: "tool_a"}},
					},
				}),
			},
		},
		{
			Role: schema.AgenticRoleTypeAssistant,
			ContentBlocks: []*schema.ContentBlock{
				schema.NewContentBlock(&schema.AssistantGenText{Text: "answer"}),
				schema.NewContentBlock(&schema.Reasoning{Text: "thinking", Signature: "sig_1"}),
				schema.NewContentBlock(&schema.FunctionToolCall{
					CallID:    "call_tool",
					Name:      "get_weather",
					Arguments: `{"city":"beijing"}`,
				}),
				schema.NewContentBlock(&schema.ServerToolCall{
					CallID: "call_server",
					Name:   string(ServerToolNameWebSearch),
					Arguments: &ServerToolCallArguments{
						WebSearch: &WebSearchArguments{Query: "golang"},
					},
				}),
			},
		},
	}

	systemBlocks, msgParams, err := toAnthropicMessages(input)
	if err != nil {
		t.Fatalf("toAnthropicMessages() error = %v", err)
	}
	if len(systemBlocks) != 1 || systemBlocks[0].Text != "follow system" {
		t.Fatalf("systemBlocks = %#v", systemBlocks)
	}
	if len(msgParams) != 2 {
		t.Fatalf("len(msgParams) = %d, want 2", len(msgParams))
	}
	if msgParams[0].Role != anthropic.MessageParamRoleUser || msgParams[1].Role != anthropic.MessageParamRoleAssistant {
		t.Fatalf("roles = [%q, %q]", msgParams[0].Role, msgParams[1].Role)
	}

	got := mustJSON(t, msgParams)
	for _, want := range []string{
		`"text":"hello"`,
		`"type":"image"`,
		`"type":"document"`,
		`"tool_use_id":"call_fn"`,
		`"tool_use_id":"call_search"`,
		`"tool_name":"tool_a"`,
		`"thinking":"thinking"`,
		`"signature":"sig_1"`,
		`"type":"tool_use"`,
		`"name":"web_search"`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("toAnthropicMessages() json = %s, want substring %s", got, want)
		}
	}
}

func TestToAnthropicMessagesErrors(t *testing.T) {
	t.Run("empty input", func(t *testing.T) {
		_, _, err := toAnthropicMessages(nil)
		if err == nil || err.Error() != "input is empty" {
			t.Fatalf("toAnthropicMessages() error = %v", err)
		}
	})

	t.Run("system after user", func(t *testing.T) {
		_, _, err := toAnthropicMessages([]*schema.AgenticMessage{
			schema.UserAgenticMessage("hello"),
			schema.SystemAgenticMessage("late system"),
		})
		if err == nil || err.Error() != "system message must appear before all non-system messages" {
			t.Fatalf("toAnthropicMessages() error = %v", err)
		}
	})
}

func TestToAgenticMessage(t *testing.T) {
	resp := &anthropic.Message{}
	if err := json.Unmarshal([]byte(`{
		"id":"msg_1",
		"type":"message",
		"role":"assistant",
		"model":"claude-sonnet-4-20250514",
		"content":[
			{"type":"text","text":"hello"},
			{"type":"thinking","thinking":"reasoning","signature":"sig_1"},
			{"type":"tool_use","id":"call_1","name":"get_weather","input":{"city":"beijing"}},
			{"type":"redacted_thinking","data":"secret"}
		],
		"stop_reason":"end_turn",
		"usage":{"input_tokens":3,"output_tokens":5}
	}`), resp); err != nil {
		t.Fatalf("json.Unmarshal(message) error = %v", err)
	}

	msg, err := toAgenticMessage(resp)
	if err != nil {
		t.Fatalf("toAgenticMessage() error = %v", err)
	}
	if msg.Role != schema.AgenticRoleTypeAssistant {
		t.Fatalf("role = %q, want assistant", msg.Role)
	}
	if len(msg.ContentBlocks) != 3 {
		t.Fatalf("len(msg.ContentBlocks) = %d, want 3", len(msg.ContentBlocks))
	}
	if msg.ContentBlocks[0].AssistantGenText == nil || msg.ContentBlocks[0].AssistantGenText.Text != "hello" {
		t.Fatalf("text block = %#v", msg.ContentBlocks[0])
	}
	if msg.ContentBlocks[1].Reasoning == nil || msg.ContentBlocks[1].Reasoning.Signature != "sig_1" {
		t.Fatalf("reasoning block = %#v", msg.ContentBlocks[1])
	}
	if msg.ContentBlocks[2].FunctionToolCall == nil || msg.ContentBlocks[2].FunctionToolCall.Arguments != `{"city":"beijing"}` {
		t.Fatalf("tool call block = %#v", msg.ContentBlocks[2])
	}
	if msg.ResponseMeta == nil || msg.ResponseMeta.ClaudeExtension == nil || msg.ResponseMeta.ClaudeExtension.ID != "msg_1" {
		t.Fatalf("response meta = %#v", msg.ResponseMeta)
	}
}

func TestToAgenticContentBlock(t *testing.T) {
	t.Run("server tool use", func(t *testing.T) {
		block, err := toAgenticContentBlock(anthropic.ServerToolUseBlock{
			ID:    "call_1",
			Name:  anthropic.ServerToolUseBlockNameWebFetch,
			Input: map[string]any{"url": "https://example.com"},
		})
		if err != nil {
			t.Fatalf("toAgenticContentBlock() error = %v", err)
		}
		if block == nil || block.ServerToolCall == nil || block.ServerToolCall.Name != string(ServerToolNameWebFetch) {
			t.Fatalf("block = %#v", block)
		}
	})

	t.Run("redacted thinking dropped", func(t *testing.T) {
		block, err := toAgenticContentBlock(anthropic.RedactedThinkingBlock{})
		if err != nil {
			t.Fatalf("toAgenticContentBlock() error = %v", err)
		}
		if block != nil {
			t.Fatalf("block = %#v, want nil", block)
		}
	})

	t.Run("invalid type", func(t *testing.T) {
		_, err := toAgenticContentBlock(123)
		if err == nil || err.Error() != "invalid output block type int" {
			t.Fatalf("toAgenticContentBlock() error = %v", err)
		}
	})
}

func TestToolMediaAndResultConversionHelpers(t *testing.T) {
	t.Run("image and file block params", func(t *testing.T) {
		imageParam, err := imageToBlockParam(&schema.UserInputImage{
			Base64Data: "aGVsbG8=",
			MIMEType:   "image/png",
		})
		if err != nil {
			t.Fatalf("imageToBlockParam() error = %v", err)
		}
		fileParam, err := documentToBlockParam(&schema.UserInputFile{
			Base64Data: "aGVsbG8=",
		})
		if err != nil {
			t.Fatalf("documentToBlockParam() error = %v", err)
		}
		got := mustJSON(t, []anthropic.ContentBlockParamUnion{imageParam, fileParam})
		for _, want := range []string{
			`"media_type":"image/png"`,
			`"type":"image"`,
			`"media_type":"application/pdf"`,
			`"type":"document"`,
		} {
			if !strings.Contains(got, want) {
				t.Fatalf("conversion json = %s, want substring %s", got, want)
			}
		}
	})

	t.Run("function tool result content image and file", func(t *testing.T) {
		imageBlock, err := functionToolResultContentToBlockParam(&schema.FunctionToolResultContentBlock{
			Type: schema.FunctionToolResultContentBlockTypeImage,
			Image: &schema.UserInputImage{
				URL: "https://example.com/image.png",
			},
		})
		if err != nil {
			t.Fatalf("functionToolResultContentToBlockParam(image) error = %v", err)
		}
		fileBlock, err := functionToolResultContentToBlockParam(&schema.FunctionToolResultContentBlock{
			Type: schema.FunctionToolResultContentBlockTypeFile,
			File: &schema.UserInputFile{
				URL: "https://example.com/doc.pdf",
			},
		})
		if err != nil {
			t.Fatalf("functionToolResultContentToBlockParam(file) error = %v", err)
		}
		got := mustJSON(t, []anthropic.ToolResultBlockParamContentUnion{imageBlock, fileBlock})
		for _, want := range []string{
			`"type":"image"`,
			`"type":"document"`,
		} {
			if !strings.Contains(got, want) {
				t.Fatalf("conversion json = %s, want substring %s", got, want)
			}
		}
	})

	t.Run("web search and web fetch result block params", func(t *testing.T) {
		searchBlock := &schema.ContentBlock{}
		setWebSearchResultCaller(searchBlock, mustWebSearchCaller(t))
		searchParam, err := webSearchToolResultToBlockParam(&WebSearchResult{
			Type: WebSearchResultTypeResult,
			Result: &WebSearchResultBlock{
				Content: []*WebSearchResultItem{
					{
						Title:            "doc",
						URL:              "https://example.com",
						EncryptedContent: "enc_1",
						PageAge:          "1 day",
					},
				},
			},
		}, "call_search", searchBlock)
		if err != nil {
			t.Fatalf("webSearchToolResultToBlockParam() error = %v", err)
		}

		fetchBlock := &schema.ContentBlock{}
		setWebFetchResultCaller(fetchBlock, mustWebFetchCaller(t))
		fetchParam, err := webFetchToolResultToBlockParam(&WebFetchResult{
			Type: WebFetchResultTypeResult,
			Result: &WebFetchResultBlock{
				URL:         "https://example.com",
				RetrievedAt: "2026-05-19T00:00:00Z",
				Content: &WebFetchDocument{
					Title: "doc",
					Source: &WebFetchDocumentSource{
						MIMEType: "text/plain",
						Data:     "hello",
					},
				},
			},
		}, "call_fetch", fetchBlock)
		if err != nil {
			t.Fatalf("webFetchToolResultToBlockParam() error = %v", err)
		}

		got := mustJSON(t, []anthropic.ContentBlockParamUnion{searchParam, fetchParam})
		for _, want := range []string{
			`"type":"web_search_tool_result"`,
			`"encrypted_content":"enc_1"`,
			`"type":"web_fetch_tool_result"`,
			`"retrieved_at":"2026-05-19T00:00:00Z"`,
			`"caller":{"type":"direct"}`,
		} {
			if !strings.Contains(got, want) {
				t.Fatalf("conversion json = %s, want substring %s", got, want)
			}
		}
	})

	t.Run("execution result block params and output helpers", func(t *testing.T) {
		codeParam, err := codeExecutionToolResultToBlockParam(&CodeExecutionResult{
			Type: CodeExecutionResultTypeResult,
			Result: &CodeExecutionResultBlock{
				Content: []*CodeExecutionOutput{{FileID: "file_1"}, nil},
				Stdout:  "stdout",
			},
		}, "call_code")
		if err != nil {
			t.Fatalf("codeExecutionToolResultToBlockParam() error = %v", err)
		}
		bashParam, err := bashCodeExecutionToolResultToBlockParam(&BashCodeExecutionResult{
			Type: BashCodeExecutionResultTypeResult,
			Result: &BashCodeExecutionResultBlock{
				Content: []*CodeExecutionOutput{{FileID: "file_2"}, nil},
				Stdout:  "ok",
			},
		}, "call_bash")
		if err != nil {
			t.Fatalf("bashCodeExecutionToolResultToBlockParam() error = %v", err)
		}
		textEditorParam, err := textEditorCodeExecutionToolResultToBlockParam(&TextEditorCodeExecutionResult{
			Type: TextEditorCodeExecutionResultTypeView,
			View: &TextEditorCodeExecutionViewResult{
				FileType:   "text",
				Content:    "hello",
				NumLines:   1,
				StartLine:  1,
				TotalLines: 1,
			},
		}, "call_editor")
		if err != nil {
			t.Fatalf("textEditorCodeExecutionToolResultToBlockParam() error = %v", err)
		}

		if got := toCodeExecutionOutputs([]anthropic.CodeExecutionOutputBlock{{FileID: "file_1"}}); len(got) != 1 || got[0].FileID != "file_1" {
			t.Fatalf("toCodeExecutionOutputs() = %#v", got)
		}
		if got := toBashCodeExecutionOutputs([]anthropic.BashCodeExecutionOutputBlock{{FileID: "file_2"}}); len(got) != 1 || got[0].FileID != "file_2" {
			t.Fatalf("toBashCodeExecutionOutputs() = %#v", got)
		}

		got := mustJSON(t, []anthropic.ContentBlockParamUnion{codeParam, bashParam, textEditorParam})
		for _, want := range []string{
			`"type":"code_execution_tool_result"`,
			`"type":"bash_code_execution_tool_result"`,
			`"type":"text_editor_code_execution_tool_result"`,
			`"file_id":"file_1"`,
			`"file_id":"file_2"`,
		} {
			if !strings.Contains(got, want) {
				t.Fatalf("conversion json = %s, want substring %s", got, want)
			}
		}
	})

	t.Run("server tool use helpers", func(t *testing.T) {
		cases := []anthropic.ServerToolUseBlock{
			{ID: "call_search", Name: anthropic.ServerToolUseBlockNameWebSearch, Input: map[string]any{"query": "golang"}},
			{ID: "call_code", Name: anthropic.ServerToolUseBlockNameCodeExecution, Input: map[string]any{"code": "print(1)"}},
			{ID: "call_bash", Name: anthropic.ServerToolUseBlockNameBashCodeExecution, Input: map[string]any{"command": "ls"}},
			{ID: "call_editor", Name: anthropic.ServerToolUseBlockNameTextEditorCodeExecution, Input: map[string]any{"command": "view", "path": "/tmp/a.txt"}},
			{ID: "call_regex", Name: anthropic.ServerToolUseBlockNameToolSearchToolRegex, Input: map[string]any{"query": "find.*"}},
		}

		for _, tc := range cases {
			block, err := serverToolUseToContentBlock(tc)
			if err != nil {
				t.Fatalf("serverToolUseToContentBlock(%q) error = %v", tc.Name, err)
			}
			if block == nil || block.ServerToolCall == nil || block.ServerToolCall.Name == "" {
				t.Fatalf("server tool block = %#v", block)
			}
		}
	})
}

func mustWebSearchCaller(t *testing.T) anthropic.WebSearchToolResultBlockCallerUnion {
	t.Helper()

	var caller anthropic.WebSearchToolResultBlockCallerUnion
	if err := json.Unmarshal([]byte(`{"type":"direct"}`), &caller); err != nil {
		t.Fatalf("json.Unmarshal(web search caller) error = %v", err)
	}
	return caller
}

func mustWebFetchCaller(t *testing.T) anthropic.WebFetchToolResultBlockCallerUnion {
	t.Helper()

	var caller anthropic.WebFetchToolResultBlockCallerUnion
	if err := json.Unmarshal([]byte(`{"type":"direct"}`), &caller); err != nil {
		t.Fatalf("json.Unmarshal(web fetch caller) error = %v", err)
	}
	return caller
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()

	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal(%T) error = %v", v, err)
	}
	return string(data)
}
