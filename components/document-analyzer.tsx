"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { FileUpload } from "@/components/file-upload"
import { NERResults } from "@/components/ner-results"
import { SummaryResults } from "@/components/summary-results"
import { RAGChat } from "@/components/rag-chat"
import { Scale, FileText, MessageSquare, Upload } from "lucide-react"

interface AnalysisResults {
  ner_results?: any
  summary_results?: any
  text_stats?: any
}

export function DocumentAnalyzer() {
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [currentDocument, setCurrentDocument] = useState<string>("")

  const handleAnalysisComplete = (results: AnalysisResults, documentText: string) => {
    setAnalysisResults(results)
    setCurrentDocument(documentText)
    setIsAnalyzing(false)
  }

  const handleAnalysisStart = () => {
    setIsAnalyzing(true)
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="flex items-center justify-center gap-3 mb-4">
          <Scale className="h-8 w-8 text-primary" />
          <h1 className="text-4xl font-bold text-foreground">Legal Document Analyzer</h1>
        </div>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Advanced AI-powered analysis for legal documents using Named Entity Recognition, Summarization, and
          Retrieval-Augmented Generation
        </p>
      </div>

      <Tabs defaultValue="upload" className="w-full">
        <TabsList className="grid w-full grid-cols-4 mb-8">
          <TabsTrigger value="upload" className="flex items-center gap-2">
            <Upload className="h-4 w-4" />
            Upload & Analyze
          </TabsTrigger>
          <TabsTrigger value="ner" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Named Entities
          </TabsTrigger>
          <TabsTrigger value="summary" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Summary
          </TabsTrigger>
          <TabsTrigger value="rag" className="flex items-center gap-2">
            <MessageSquare className="h-4 w-4" />
            Ask Questions
          </TabsTrigger>
        </TabsList>

        <TabsContent value="upload" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Document Upload & Analysis</CardTitle>
              <CardDescription>
                Upload a legal document to automatically perform Named Entity Recognition and Summarization
              </CardDescription>
            </CardHeader>
            <CardContent>
              <FileUpload
                onAnalysisComplete={handleAnalysisComplete}
                onAnalysisStart={handleAnalysisStart}
                isAnalyzing={isAnalyzing}
              />
            </CardContent>
          </Card>

          {/* Analysis Status */}
          {analysisResults && (
            <Card>
              <CardHeader>
                <CardTitle>Analysis Complete</CardTitle>
                <CardDescription>Your document has been processed. View results in the tabs above.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-muted rounded-lg">
                    <div className="text-2xl font-bold text-primary">
                      {analysisResults.text_stats?.character_count || 0}
                    </div>
                    <div className="text-sm text-muted-foreground">Characters</div>
                  </div>
                  <div className="text-center p-4 bg-muted rounded-lg">
                    <div className="text-2xl font-bold text-primary">{analysisResults.text_stats?.word_count || 0}</div>
                    <div className="text-sm text-muted-foreground">Words</div>
                  </div>
                  <div className="text-center p-4 bg-muted rounded-lg">
                    <div className="text-2xl font-bold text-primary">
                      {analysisResults.ner_results?.total_entities || 0}
                    </div>
                    <div className="text-sm text-muted-foreground">Entities Found</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="ner">
          <NERResults results={analysisResults?.ner_results} />
        </TabsContent>

        <TabsContent value="summary">
          <SummaryResults results={analysisResults?.summary_results} documentText={currentDocument} />
        </TabsContent>

        <TabsContent value="rag">
          <RAGChat documentText={currentDocument} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
