"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import type { UnifiedPredictionResult } from "@/services/unified-api"

interface ModernPredictionResultsProps {
  results: UnifiedPredictionResult
}

export function ModernPredictionResults({ results }: ModernPredictionResultsProps) {
  const isMalignant = results.prediction === 'malignant'
  const confidencePercent = Math.round(results.confidence * 100)

  return (
    <Card className="w-full h-full flex flex-col">
      <CardContent className="p-6 flex-grow flex flex-col justify-center items-center text-center">
        <h3 className="text-sm font-semibold text-muted-foreground mb-2">PREDICTION</h3>
        <p className={`text-4xl font-bold ${isMalignant ? 'text-red-500' : 'text-green-500'}`}>
          {isMalignant ? "Malignant" : "Benign"}
        </p>
        <p className="text-lg text-muted-foreground mt-1">{confidencePercent}% Confidence</p>

        {results.probabilities && (
          <div className="w-full space-y-4 mt-8">
            <div>
              <div className="flex justify-between mb-1 text-sm">
                <span className="font-medium">Benign</span>
                <span>{Math.round(results.probabilities.benign * 100)}%</span>
              </div>
              <Progress value={results.probabilities.benign * 100} className="h-2" indicatorClassName="bg-green-500" />
            </div>
            <div>
              <div className="flex justify-between mb-1 text-sm">
                <span className="font-medium">Malignant</span>
                <span>{Math.round(results.probabilities.malignant * 100)}%</span>
              </div>
              <Progress value={results.probabilities.malignant * 100} className="h-2" indicatorClassName="bg-red-500" />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
