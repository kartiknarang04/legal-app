"use client";

import type React from "react";

import { useState, useRef, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  MessageSquare,
  Send,
  Loader2,
  Bot,
  User,
  FileText,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Database,
} from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import axios from "axios";

interface Message {
  id: string;
  type: "user" | "assistant";
  content: string;
  timestamp: Date;
  contexts?: Array<{
    title: string;
    score: number;
    chunk_index: number;
    legal_terms: string[];
    relevant_passages: string[];
  }>;
  metadata?: {
    model_used: string;
    num_contexts: number;
  };
}

interface RAGChatProps {
  documentText: string;
}

export function RAGChat({ documentText }: RAGChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [ragStatus, setRagStatus] = useState<any>(null);
  const [expandedContexts, setExpandedContexts] = useState<
    Record<string, boolean>
  >({});
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Check RAG system status on mount
  useEffect(() => {
    checkRagStatus();
  }, []);

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const checkRagStatus = async () => {
    try {
      const response = await axios.get(
        `${process.env.NEXT_PUBLIC_API_URL}/rag/status`
      );
      setRagStatus(response.data);
    } catch (error) {
      console.error("Failed to check RAG status:", error);
    }
  };

  const addDocumentToRAG = async () => {
    if (!documentText) return;

    try {
      setIsLoading(true);
      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/rag/add_document`,
        {
          text: documentText,
        }
      );

      if (response.data.session_id) {
        setCurrentSessionId(response.data.session_id);
      }

      await checkRagStatus();

      // Add system message
      const systemMessage: Message = {
        id: Date.now().toString(),
        type: "assistant",
        content: `Document has been added to the knowledge base (Session: ${response.data.session_id}). You can now ask questions about it!`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, systemMessage]);
    } catch (error) {
      console.error("Failed to add document to RAG:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: inputValue,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/rag/query`,
        {
          question: inputValue,
          session_id: currentSessionId,
          top_k: 5,
          use_rerank: true,
        }
      );

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content: response.data.answer,
        timestamp: new Date(),
        contexts: response.data.contexts,
        metadata: response.data.metadata,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error: any) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: "assistant",
        content: `Error: ${
          error.response?.data?.detail ||
          "Failed to get response from RAG system"
        }`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const toggleContextExpansion = (messageId: string) => {
    setExpandedContexts((prev) => ({
      ...prev,
      [messageId]: !prev[messageId],
    }));
  };

  return (
    <div className="space-y-6">
      {/* RAG Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Knowledge Base Status
          </CardTitle>
          <CardDescription>
            Current status of the RAG (Retrieval-Augmented Generation) system
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div
                  className={`w-3 h-3 rounded-full ${
                    ragStatus?.available ? "bg-green-500" : "bg-red-500"
                  }`}
                />
                <span className="text-sm font-medium">
                  {ragStatus?.available ? "Available" : "Not Available"}
                </span>
              </div>
              <Badge variant="secondary">
                {ragStatus?.documents || 0} documents indexed
              </Badge>
            </div>
            {documentText && (
              <Button onClick={addDocumentToRAG} disabled={isLoading} size="sm">
                {isLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Adding...
                  </>
                ) : (
                  "Add Current Document"
                )}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Chat Interface */}
      <Card className="h-[600px] flex flex-col">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            Ask Questions About Your Documents
          </CardTitle>
          <CardDescription>
            Use natural language to query your legal documents
          </CardDescription>
        </CardHeader>

        <CardContent className="flex-1 flex flex-col p-0">
          {/* Messages */}
          <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
            {messages.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <MessageSquare className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p className="mb-2">No messages yet</p>
                <p className="text-sm">
                  {ragStatus?.documents > 0
                    ? "Ask questions about your documents!"
                    : "Add documents to the knowledge base to start asking questions"}
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((message) => (
                  <div key={message.id} className="space-y-2">
                    <div
                      className={`flex gap-3 ${
                        message.type === "user"
                          ? "justify-end"
                          : "justify-start"
                      }`}
                    >
                      <div
                        className={`flex gap-3 max-w-[80%] ${
                          message.type === "user"
                            ? "flex-row-reverse"
                            : "flex-row"
                        }`}
                      >
                        <div
                          className={`w-8 h-8 rounded-full flex items-center justify-center ${
                            message.type === "user"
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted text-muted-foreground"
                          }`}
                        >
                          {message.type === "user" ? (
                            <User className="h-4 w-4" />
                          ) : (
                            <Bot className="h-4 w-4" />
                          )}
                        </div>
                        <div
                          className={`rounded-lg p-3 ${
                            message.type === "user"
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted"
                          }`}
                        >
                          <p className="text-sm leading-relaxed whitespace-pre-wrap">
                            {message.content}
                          </p>
                          <div className="text-xs opacity-70 mt-2">
                            {message.timestamp.toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Context Sources */}
                    {message.contexts && message.contexts.length > 0 && (
                      <div className="ml-11 space-y-2">
                        <Collapsible>
                          <CollapsibleTrigger
                            onClick={() => toggleContextExpansion(message.id)}
                            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
                          >
                            <FileText className="h-4 w-4" />
                            <span>Sources ({message.contexts.length})</span>
                            {expandedContexts[message.id] ? (
                              <ChevronUp className="h-4 w-4" />
                            ) : (
                              <ChevronDown className="h-4 w-4" />
                            )}
                          </CollapsibleTrigger>
                          <CollapsibleContent className="space-y-2 mt-2">
                            {message.contexts.map((context, index) => (
                              <div
                                key={index}
                                className="p-3 bg-muted/50 rounded-lg border"
                              >
                                <div className="flex items-center justify-between mb-2">
                                  <div className="flex items-center gap-2">
                                    <Badge
                                      variant="outline"
                                      className="text-xs"
                                    >
                                      Chunk {context.chunk_index}
                                    </Badge>
                                    <span className="text-sm font-medium">
                                      {context.title || "Document"}
                                    </span>
                                  </div>
                                  <Badge
                                    variant="secondary"
                                    className="text-xs"
                                  >
                                    Score:{" "}
                                    {(
                                      context.score ||
                                      context.relevance_score ||
                                      0
                                    ).toFixed(3)}
                                  </Badge>
                                </div>

                                {context.legal_terms &&
                                  context.legal_terms.length > 0 && (
                                    <div className="mb-2">
                                      <div className="text-xs text-muted-foreground mb-1">
                                        Legal Terms:
                                      </div>
                                      <div className="flex flex-wrap gap-1">
                                        {context.legal_terms
                                          .slice(0, 5)
                                          .map((term, termIndex) => (
                                            <Badge
                                              key={termIndex}
                                              variant="outline"
                                              className="text-xs"
                                            >
                                              {term}
                                            </Badge>
                                          ))}
                                      </div>
                                    </div>
                                  )}

                                {context.relevant_passages &&
                                  context.relevant_passages.length > 0 && (
                                    <div>
                                      <div className="text-xs text-muted-foreground mb-1">
                                        Relevant Passages:
                                      </div>
                                      <div className="text-xs text-muted-foreground space-y-1">
                                        {context.relevant_passages.map(
                                          (passage, passageIndex) => (
                                            <p
                                              key={passageIndex}
                                              className="italic"
                                            >
                                              "{passage.substring(0, 150)}..."
                                            </p>
                                          )
                                        )}
                                      </div>
                                    </div>
                                  )}
                              </div>
                            ))}
                          </CollapsibleContent>
                        </Collapsible>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>

          <Separator />

          {/* Input */}
          <div className="p-4">
            {!ragStatus?.available && (
              <Alert className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  RAG system is not available. Please check if the backend is
                  running.
                </AlertDescription>
              </Alert>
            )}

            <div className="flex gap-2">
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  ragStatus?.documents > 0
                    ? "Ask a question about your documents..."
                    : "Add documents to start asking questions..."
                }
                disabled={
                  isLoading ||
                  !ragStatus?.available ||
                  ragStatus?.documents === 0
                }
                className="flex-1"
              />
              <Button
                onClick={sendMessage}
                disabled={
                  isLoading ||
                  !inputValue.trim() ||
                  !ragStatus?.available ||
                  ragStatus?.documents === 0
                }
                size="icon"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
