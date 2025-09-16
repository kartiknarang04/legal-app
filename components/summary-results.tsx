"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  AlertCircle,
  FileText,
  Loader2,
  Sparkles,
  Copy,
  Check,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import axios from "axios";

interface SummaryResultsProps {
  results?: {
    summary: string;
    extractive_summary?: string;
    extracted_sentences: string[];
    sentence_indices: number[];
    refined: boolean;
    error?: string;
  };
  documentText: string;
}

export function SummaryResults({ results, documentText }: SummaryResultsProps) {
  const [summaryLength, setSummaryLength] = useState([5]);
  const [useGroqRefinement, setUseGroqRefinement] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [customResults, setCustomResults] = useState<any>(null);
  const [copied, setCopied] = useState(false);

  const generateCustomSummary = async () => {
    if (!documentText) {
      return;
    }

    try {
      setIsGenerating(true);

      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/analyze`,
        {
          text: documentText,
          summary_length: summaryLength[0],
          use_groq_refinement: useGroqRefinement,
        }
      );

      if (response.data.success) {
        setCustomResults(response.data.analysis.summary_results);
      }
    } catch (error) {
      console.error("Error generating custom summary:", error);
    } finally {
      setIsGenerating(false);
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error("Failed to copy text:", error);
    }
  };

  const displayResults = customResults || results;

  if (!results && !documentText) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Document Summarization</CardTitle>
          <CardDescription>
            Upload and analyze a document to see the generated summary
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No analysis results available</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (displayResults?.error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Document Summarization</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{displayResults.error}</AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary Controls */}
      <Card>
        <CardHeader>
          <CardTitle>Summary Configuration</CardTitle>
          <CardDescription>
            Customize the summary length and refinement options
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3">
            <Label>Summary Length: {summaryLength[0]} sentences</Label>
            <Slider
              value={summaryLength}
              onValueChange={setSummaryLength}
              max={20}
              min={3}
              step={1}
              className="w-full"
            />
          </div>

          <div className="flex items-center space-x-2">
            <Switch
              id="groq-refinement"
              checked={useGroqRefinement}
              onCheckedChange={setUseGroqRefinement}
            />
            <Label
              htmlFor="groq-refinement"
              className="flex items-center gap-2"
            >
              <Sparkles className="h-4 w-4" />
              Use AI Refinement (Groq)
            </Label>
          </div>

          <Button
            onClick={generateCustomSummary}
            disabled={isGenerating || !documentText}
            className="w-full"
          >
            {isGenerating ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Generating Summary...
              </>
            ) : (
              "Generate Custom Summary"
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Summary Results */}
      {displayResults && (
        <>
          {/* Main Summary */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    {displayResults.refined && (
                      <Sparkles className="h-5 w-5 text-primary" />
                    )}
                    {displayResults.refined
                      ? "AI-Refined Summary"
                      : "Extractive Summary"}
                  </CardTitle>
                  <CardDescription>
                    {displayResults.refined
                      ? "Enhanced summary with improved flow and readability"
                      : "Key sentences extracted from the original document"}
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => copyToClipboard(displayResults.summary)}
                >
                  {copied ? (
                    <Check className="h-4 w-4" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-64">
                <div className="prose prose-sm max-w-none">
                  <p className="text-foreground leading-relaxed whitespace-pre-wrap">
                    {displayResults.summary}
                  </p>
                </div>
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Summary Statistics */}
          <Card>
            <CardHeader>
              <CardTitle>Summary Statistics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="text-2xl font-bold text-primary">
                    {displayResults.extracted_sentences?.length || 0}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Sentences Extracted
                  </div>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="text-2xl font-bold text-primary">
                    {displayResults.summary?.split(" ").length || 0}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Words in Summary
                  </div>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="text-2xl font-bold text-primary">
                    {documentText
                      ? Math.round(
                          ((displayResults.summary?.length || 0) /
                            documentText.length) *
                            100
                        )
                      : 0}
                    %
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Compression Ratio
                  </div>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <Badge
                    variant={displayResults.refined ? "default" : "secondary"}
                    className="text-sm"
                  >
                    {displayResults.refined ? "AI Enhanced" : "Extractive"}
                  </Badge>
                  <div className="text-sm text-muted-foreground mt-1">
                    Processing Type
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Extracted Sentences */}
          {displayResults.extracted_sentences &&
            displayResults.extracted_sentences.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Key Sentences</CardTitle>
                  <CardDescription>
                    Individual sentences that were selected for the summary
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-96">
                    <div className="space-y-4">
                      {displayResults.extracted_sentences.map(
                        (sentence: string, index: number) => (
                          <div key={index} className="p-4 bg-muted rounded-lg">
                            <div className="flex items-start gap-3">
                              <Badge variant="outline" className="mt-1 text-xs">
                                #
                                {displayResults.sentence_indices?.[index] ||
                                  index + 1}
                              </Badge>
                              <p className="text-sm leading-relaxed flex-1">
                                {sentence}
                              </p>
                            </div>
                          </div>
                        )
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            )}

          {/* Comparison (if both extractive and refined exist) */}
          {displayResults.extractive_summary && displayResults.refined && (
            <Card>
              <CardHeader>
                <CardTitle>Summary Comparison</CardTitle>
                <CardDescription>
                  Compare the extractive summary with the AI-refined version
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium mb-3 flex items-center gap-2">
                      <FileText className="h-4 w-4" />
                      Extractive Summary
                    </h4>
                    <ScrollArea className="h-48 p-3 bg-muted rounded-lg">
                      <p className="text-sm leading-relaxed">
                        {displayResults.extractive_summary}
                      </p>
                    </ScrollArea>
                  </div>
                  <div>
                    <h4 className="font-medium mb-3 flex items-center gap-2">
                      <Sparkles className="h-4 w-4" />
                      AI-Refined Summary
                    </h4>
                    <ScrollArea className="h-48 p-3 bg-muted rounded-lg">
                      <p className="text-sm leading-relaxed">
                        {displayResults.summary}
                      </p>
                    </ScrollArea>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
