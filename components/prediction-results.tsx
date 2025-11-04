import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import type { UnifiedPredictionResult } from "@/services/unified-api"

interface PredictionResultsProps {
    results: UnifiedPredictionResult | null
    error?: string | null
}

export function PredictionResults({ results, error }: PredictionResultsProps) {
    if (error) {
        return (
            <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
            </Alert>
        )
    }

    if (!results) return null

    const confidencePercent = Math.round(results.confidence * 100)
    const isMalignant = results.prediction === 'malignant'

    return (
        <Card className="w-full">
            <CardHeader>
                <CardTitle className={isMalignant ? "text-red-500" : "text-green-500"}>
                    {isMalignant ? "Malignant" : "Benign"} ({confidencePercent}% confidence)
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="space-y-4">
                    <div>
                        <div className="flex justify-between mb-1">
                            <span className="text-sm font-medium">Benign</span>
                            <span className="text-sm font-medium">
                                {Math.round(results.probabilities.benign * 100)}%
                            </span>
                        </div>
                        <Progress
                            value={results.probabilities.benign * 100}
                            className="bg-gray-200"
                        />
                    </div>
                    <div>
                        <div className="flex justify-between mb-1">
                            <span className="text-sm font-medium">Malignant</span>
                            <span className="text-sm font-medium">
                                {Math.round(results.probabilities.malignant * 100)}%
                            </span>
                        </div>
                        <Progress
                            value={results.probabilities.malignant * 100}
                            className="bg-gray-200"
                        />
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}