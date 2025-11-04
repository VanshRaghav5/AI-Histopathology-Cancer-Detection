"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { AlertCircle, CheckCircle, Eye, EyeOff, Layers, ImageIcon } from "lucide-react"
import { useState } from "react"

interface ResultsDisplayProps {
  prediction: "Cancerous" | "Non-Cancerous"
  confidence: number
  originalImage: string
  heatmapImage?: string
  processingTime?: number
}

export function ResultsDisplay({
  prediction,
  confidence,
  originalImage,
  heatmapImage,
  processingTime,
}: ResultsDisplayProps) {
  const [showHeatmap, setShowHeatmap] = useState(false)
  const [overlayMode, setOverlayMode] = useState<"side-by-side" | "overlay">("side-by-side")

  const isCancerous = prediction === "Cancerous"
  const confidencePercentage = Math.round(confidence * 100)

  return (
    <div className="w-full max-w-4xl mx-auto space-y-4 sm:space-y-6">
      {/* Results Summary Card */}
      <Card>
        <CardHeader>
          <CardTitle className="font-serif text-xl sm:text-2xl flex items-center gap-3">
            {isCancerous ? (
              <AlertCircle className="h-5 w-5 sm:h-6 sm:w-6 text-destructive" />
            ) : (
              <CheckCircle className="h-5 w-5 sm:h-6 sm:w-6 text-primary" />
            )}
            Analysis Results
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <p className="font-sans text-sm text-muted-foreground mb-1">Diagnosis</p>
              <Badge variant={isCancerous ? "destructive" : "default"} className="text-sm px-3 py-1">
                {prediction}
              </Badge>
            </div>
            <div className="flex flex-col sm:flex-row gap-4 sm:gap-8">
              <div className="text-left sm:text-right">
                <p className="font-sans text-sm text-muted-foreground mb-1">Confidence Level</p>
                <div className="flex items-center gap-2">
                  <div className="w-20 sm:w-24 h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${isCancerous ? "bg-destructive" : "bg-primary"}`}
                      style={{ width: `${confidencePercentage}%` }}
                    />
                  </div>
                  <span className="font-sans font-semibold text-sm">{confidencePercentage}%</span>
                </div>
              </div>
              {processingTime && (
                <div className="text-left sm:text-right">
                  <p className="font-sans text-sm text-muted-foreground mb-1">Processing Time</p>
                  <span className="font-sans font-semibold text-sm">{processingTime.toFixed(2)}s</span>
                </div>
              )}
            </div>
          </div>

          <div className="p-3 sm:p-4 bg-muted/50 rounded-lg">
            <p className="font-sans text-xs sm:text-sm text-foreground">
              <strong>Clinical Note:</strong> This AI analysis is for research and educational purposes. Always consult
              with qualified medical professionals for clinical diagnosis and treatment decisions.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Image Comparison Card */}
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <CardTitle className="font-serif text-lg sm:text-xl">Image Analysis</CardTitle>
            <div className="flex flex-wrap items-center gap-2">
              {heatmapImage && (
                <>
                  <Button
                    variant={overlayMode === "side-by-side" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setOverlayMode("side-by-side")}
                    className="flex items-center gap-2 text-xs sm:text-sm"
                  >
                    <ImageIcon className="h-3 w-3 sm:h-4 sm:w-4" />
                    <span className="hidden sm:inline">Side by Side</span>
                    <span className="sm:hidden">Side</span>
                  </Button>
                  <Button
                    variant={overlayMode === "overlay" ? "default" : "outline"}
                    size="sm"
                    onClick={() => setOverlayMode("overlay")}
                    className="flex items-center gap-2 text-xs sm:text-sm"
                  >
                    <Layers className="h-3 w-3 sm:h-4 sm:w-4" />
                    Overlay
                  </Button>
                </>
              )}
              {heatmapImage && overlayMode === "overlay" && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  className="flex items-center gap-2 text-xs sm:text-sm"
                >
                  {showHeatmap ? (
                    <EyeOff className="h-3 w-3 sm:h-4 sm:w-4" />
                  ) : (
                    <Eye className="h-3 w-3 sm:h-4 sm:w-4" />
                  )}
                  <span className="hidden sm:inline">{showHeatmap ? "Hide Heatmap" : "Show Heatmap"}</span>
                  <span className="sm:hidden">{showHeatmap ? "Hide" : "Show"}</span>
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {overlayMode === "side-by-side" && heatmapImage ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
              <div className="space-y-2">
                <h3 className="font-sans font-medium text-foreground text-sm sm:text-base">Original Image</h3>
                <div className="relative rounded-lg overflow-hidden border border-border">
                  <img
                    src={originalImage || "/placeholder.svg"}
                    alt="Original histopathology image"
                    className="w-full h-48 sm:h-64 object-cover"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <h3 className="font-sans font-medium text-foreground text-sm sm:text-base">AI Attention Heatmap</h3>
                <div className="relative rounded-lg overflow-hidden border border-border">
                  <img
                    src={heatmapImage || "/placeholder.svg"}
                    alt="AI attention heatmap"
                    className="w-full h-48 sm:h-64 object-cover"
                  />
                  <div className="absolute bottom-2 left-2 bg-background/90 px-2 py-1 rounded text-xs font-sans">
                    Red areas indicate higher AI attention
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="space-y-2">
                <h3 className="font-sans font-medium text-foreground text-sm sm:text-base">
                  {showHeatmap && heatmapImage ? "Overlay View with Heatmap" : "Original Image"}
                </h3>
                <div className="relative rounded-lg overflow-hidden border border-border">
                  <img
                    src={originalImage || "/placeholder.svg"}
                    alt="Original histopathology image"
                    className="w-full h-56 sm:h-64 md:h-80 object-cover"
                  />
                  {showHeatmap && heatmapImage && (
                    <img
                      src={heatmapImage || "/placeholder.svg"}
                      alt="AI attention heatmap overlay"
                      className="absolute inset-0 w-full h-full object-cover opacity-60 mix-blend-multiply"
                    />
                  )}
                  {showHeatmap && heatmapImage && (
                    <div className="absolute bottom-2 left-2 bg-background/90 px-2 py-1 rounded text-xs font-sans">
                      Heatmap overlay active - Red areas show AI focus
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {heatmapImage && (
            <div className="mt-4 p-3 sm:p-4 bg-accent/10 rounded-lg">
              <p className="font-sans text-xs sm:text-sm text-foreground">
                <strong>Heatmap Explanation:</strong> The heatmap visualization shows areas where the AI model focused
                its attention during analysis. Warmer colors (red/orange) indicate regions that contributed more
                significantly to the final prediction. Use the view modes above to compare original and heatmap images
                side-by-side or as an overlay.
              </p>
            </div>
          )}

          {!heatmapImage && (
            <div className="mt-4 p-3 sm:p-4 bg-muted/50 rounded-lg">
              <p className="font-sans text-xs sm:text-sm text-muted-foreground">
                Heatmap visualization not available for this analysis.
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
