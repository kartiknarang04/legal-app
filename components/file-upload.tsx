"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Loader2, Upload, FileText, AlertCircle } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import axios from "axios";

interface FileUploadProps {
  onAnalysisComplete: (results: any, documentText: string) => void;
  onAnalysisStart: () => void;
  isAnalyzing: boolean;
}

export function FileUpload({
  onAnalysisComplete,
  onAnalysisStart,
  isAnalyzing,
}: FileUploadProps) {
  const [textInput, setTextInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedFile(file);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/plain": [".txt"],
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        [".docx"],
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const analyzeDocument = async (text: string) => {
    try {
      onAnalysisStart();
      setError(null);

      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/analyze`,
        {
          text: text,
          summary_length: 5,
          use_groq_refinement: true,
        }
      );

      if (response.data.success) {
        onAnalysisComplete(response.data.analysis, text);

        // Also add document to RAG system
        try {
          await axios.post(
            `${process.env.NEXT_PUBLIC_API_URL}/rag/add_document`,
            {
              text: text,
            }
          );
        } catch (ragError) {
          console.warn("Failed to add document to RAG system:", ragError);
        }
      } else {
        setError("Analysis failed. Please try again.");
      }
    } catch (err: any) {
      console.error("Analysis error:", err);
      setError(
        err.response?.data?.detail ||
          "Failed to analyze document. Please check if the backend is running."
      );
    }
  };

  const handleFileUpload = async () => {
    if (!uploadedFile) return;

    try {
      const formData = new FormData();
      formData.append("file", uploadedFile);

      onAnalysisStart();
      setError(null);

      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/upload`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (response.data.success) {
        // For file upload, we need to extract the text from the response
        // This is a simplified approach - in production, you'd handle this better
        const fileReader = new FileReader();
        fileReader.onload = (e) => {
          const text = e.target?.result as string;
          onAnalysisComplete(response.data.analysis, text);
        };
        fileReader.readAsText(uploadedFile);
      } else {
        setError("File upload failed. Please try again.");
      }
    } catch (err: any) {
      console.error("Upload error:", err);
      setError(
        err.response?.data?.detail ||
          "Failed to upload file. Please check if the backend is running."
      );
    }
  };

  const handleTextAnalysis = () => {
    if (!textInput.trim()) {
      setError("Please enter some text to analyze.");
      return;
    }
    analyzeDocument(textInput);
  };

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* File Upload */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <Label className="text-base font-medium">Upload Document</Label>
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? "border-primary bg-primary/5"
                  : "border-muted-foreground/25 hover:border-primary/50"
              }`}
            >
              <input {...getInputProps()} />
              <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              {isDragActive ? (
                <p className="text-primary">Drop the file here...</p>
              ) : (
                <div>
                  <p className="text-foreground font-medium mb-2">
                    Drag & drop a legal document here, or click to select
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Supports .txt, .pdf, .docx files (max 10MB)
                  </p>
                </div>
              )}
            </div>

            {uploadedFile && (
              <div className="flex items-center justify-between p-3 bg-muted rounded-lg">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  <span className="text-sm font-medium">
                    {uploadedFile.name}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    ({(uploadedFile.size / 1024).toFixed(1)} KB)
                  </span>
                </div>
                <Button
                  onClick={handleFileUpload}
                  disabled={isAnalyzing}
                  size="sm"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Analyze File"
                  )}
                </Button>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Text Input */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <Label htmlFor="text-input" className="text-base font-medium">
              Or Paste Text Directly
            </Label>
            <Textarea
              id="text-input"
              placeholder="Paste your legal document text here..."
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              className="min-h-[200px] resize-none"
            />
            <Button
              onClick={handleTextAnalysis}
              disabled={isAnalyzing || !textInput.trim()}
              className="w-full"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Analyzing Document...
                </>
              ) : (
                "Analyze Text"
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
