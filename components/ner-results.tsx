"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { AlertCircle, Users, Building, Gavel, FileText } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface NERResultsProps {
  results?: {
    entities: Array<{
      text: string
      label: string
      start: number
      end: number
    }>
    entity_counts: Record<
      string,
      {
        entities: string[]
        count: number
      }
    >
    total_entities: number
    unique_labels: string[]
    error?: string
  }
}

const entityIcons: Record<string, any> = {
  PERSON: Users,
  ORG: Building,
  PRECEDENT: Gavel,
  STATUTE: FileText,
  COURT: Building,
  JUDGE: Users,
  LAWYER: Users,
  CASE: FileText,
  DATE: FileText,
  MONEY: FileText,
  GPE: Building, // Geopolitical entity
  LAW: FileText,
}

const entityColors: Record<string, string> = {
  PERSON: "bg-blue-100 text-blue-800 border-blue-200",
  ORG: "bg-green-100 text-green-800 border-green-200",
  PRECEDENT: "bg-purple-100 text-purple-800 border-purple-200",
  STATUTE: "bg-orange-100 text-orange-800 border-orange-200",
  COURT: "bg-red-100 text-red-800 border-red-200",
  JUDGE: "bg-indigo-100 text-indigo-800 border-indigo-200",
  LAWYER: "bg-cyan-100 text-cyan-800 border-cyan-200",
  CASE: "bg-yellow-100 text-yellow-800 border-yellow-200",
  DATE: "bg-gray-100 text-gray-800 border-gray-200",
  MONEY: "bg-emerald-100 text-emerald-800 border-emerald-200",
  GPE: "bg-pink-100 text-pink-800 border-pink-200",
  LAW: "bg-violet-100 text-violet-800 border-violet-200",
}

export function NERResults({ results }: NERResultsProps) {
  if (!results) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Named Entity Recognition</CardTitle>
          <CardDescription>Upload and analyze a document to see extracted entities</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No analysis results available</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (results.error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Named Entity Recognition</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{results.error}</AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <Card>
        <CardHeader>
          <CardTitle>Named Entity Recognition Results</CardTitle>
          <CardDescription>
            Extracted {results.total_entities} entities across {results.unique_labels.length} categories
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-primary">{results.total_entities}</div>
              <div className="text-sm text-muted-foreground">Total Entities</div>
            </div>
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-primary">{results.unique_labels.length}</div>
              <div className="text-sm text-muted-foreground">Entity Types</div>
            </div>
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-primary">
                {Object.values(results.entity_counts).reduce((sum, cat) => sum + cat.entities.length, 0)}
              </div>
              <div className="text-sm text-muted-foreground">Unique Entities</div>
            </div>
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-primary">
                {Math.round(
                  (results.total_entities /
                    Object.values(results.entity_counts).reduce((sum, cat) => sum + cat.entities.length, 0)) *
                    100,
                ) || 0}
                %
              </div>
              <div className="text-sm text-muted-foreground">Repetition Rate</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Entity Categories */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {Object.entries(results.entity_counts).map(([label, data]) => {
          const Icon = entityIcons[label] || FileText
          const colorClass = entityColors[label] || "bg-gray-100 text-gray-800 border-gray-200"

          return (
            <Card key={label}>
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2">
                  <Icon className="h-5 w-5" />
                  <CardTitle className="text-lg">{label}</CardTitle>
                  <Badge variant="secondary" className="ml-auto">
                    {data.count} unique
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-32">
                  <div className="flex flex-wrap gap-2">
                    {data.entities.map((entity, index) => (
                      <Badge key={index} variant="outline" className={`${colorClass} text-xs`}>
                        {entity}
                      </Badge>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* All Entities List */}
      <Card>
        <CardHeader>
          <CardTitle>All Extracted Entities</CardTitle>
          <CardDescription>Complete list of entities found in the document with their positions</CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-96">
            <div className="space-y-2">
              {results.entities.map((entity, index) => {
                const colorClass = entityColors[entity.label] || "bg-gray-100 text-gray-800 border-gray-200"

                return (
                  <div key={index} className="flex items-center justify-between p-3 bg-muted rounded-lg">
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className={`${colorClass} text-xs`}>
                        {entity.label}
                      </Badge>
                      <span className="font-medium">{entity.text}</span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Position: {entity.start}-{entity.end}
                    </div>
                  </div>
                )
              })}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  )
}
