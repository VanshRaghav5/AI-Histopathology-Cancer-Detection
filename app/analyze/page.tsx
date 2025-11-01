"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { createClient } from "@/lib/supabase/client"
import { ImageUpload } from "@/components/image-upload"
import { ResultsDisplay } from "@/components/results-display"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Brain, Microscope, AlertCircle, RefreshCw, LogOut, BarChart3 } from "lucide-react"
import { histopathologyAPI, type PredictionResult } from "@/services/api"
import Link from "next/link"

interface AnalysisResult extends PredictionResult {
  originalImage: string
}

export default function AnalyzePage() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline" | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [user, setUser] = useState<any>(null)
  const router = useRouter()
  const supabase = createClient()

  useEffect(() => {
    const checkAuth = async () => {
      const {
        data: { user: currentUser },
      } = await supabase.auth.getUser()

      if (!currentUser) {
        router.push("/auth/login")
        return
      }

      setUser(currentUser)
      setIsLoading(false)
    }

    checkAuth()
  }, [])

  const checkApiHealth = async () => {
    setApiStatus("checking")
    try {
      const isHealthy = await histopathologyAPI.healthCheck()
      setApiStatus(isHealthy ? "online" : "offline")
    } catch {
      setApiStatus("offline")
    }
  }

  const handleImageUpload = async (file: File) => {
    setIsAnalyzing(true)
    setResult(null)
    setError(null)

    // Convert file to base64 for display
    const reader = new FileReader()
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string)
    }
    reader.readAsDataURL(file)

    try {
      console.log("[v0] Sending image to API:", file.name, file.size)
      const apiResult = await histopathologyAPI.predictCancer(file)
      console.log("[v0] API Response:", apiResult)

      if (!apiResult || typeof apiResult !== "object") {
        throw new Error("Invalid API response format")
      }

      const confidence =
        typeof apiResult.confidence === "number" && !isNaN(apiResult.confidence) ? apiResult.confidence : 0

      const prediction =
        apiResult.prediction === "Cancerous" || apiResult.prediction === "Non-Cancerous"
          ? apiResult.prediction
          : "Non-Cancerous"

      const analysisResult: AnalysisResult = {
        prediction,
        confidence,
        originalImage: URL.createObjectURL(file),
        heatmap_url:
          apiResult.heatmap_url ||
          `/placeholder.svg?height=400&width=400&query=medical heatmap overlay showing AI attention areas in red and orange colors`,
        processing_time: typeof apiResult.processing_time === "number" ? apiResult.processing_time : undefined,
      }

      console.log("[v0] Processed result:", analysisResult)
      setResult(analysisResult)

      if (user) {
        try {
          await supabase.from("analysis_history").insert({
            user_id: user.id,
            prediction,
            confidence,
            image_url: uploadedImage,
            processing_time: analysisResult.processing_time,
          })
        } catch (dbError) {
          console.error("Error saving to database:", dbError)
          // Don't fail the analysis if database save fails
        }
      }
    } catch (err) {
      console.error("[v0] Analysis failed:", err)

      if (err && typeof err === "object" && "message" in err) {
        const errorMessage = (err as Error).message

        if (errorMessage.includes("API endpoint returned HTML")) {
          setError(`API Configuration Error: The API URL appears to be incorrect. Please check that:
          • Your FastAPI server is running on the correct port
          • The NEXT_PUBLIC_API_URL environment variable points to your FastAPI server (e.g., http://localhost:8000)
          • Your FastAPI server has CORS enabled for this frontend`)
        } else if (errorMessage.includes("API endpoint not found")) {
          setError(`API Endpoint Missing: The /predict endpoint was not found. Please ensure:
          • Your FastAPI server is running
          • The /predict endpoint is properly implemented
          • The server is accessible at the configured URL`)
        } else if (errorMessage.includes("Internal server error")) {
          setError("Server Error: The AI model encountered an error. Check your FastAPI server logs for details.")
        } else if (errorMessage.includes("Network Error") || errorMessage.includes("timeout")) {
          setError(`Connection Error: Cannot reach the API server. Please verify:
          • Your FastAPI server is running
          • The API URL is correct: ${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}
          • No firewall is blocking the connection`)
        } else {
          setError(errorMessage)
        }
      } else {
        setError(
          "Unexpected error occurred. Please check the console for details and ensure your API server is running.",
        )
      }

      console.log("[v0] API call failed, not using demo mode")
    } finally {
      setIsAnalyzing(false)
    }
  }

  const resetAnalysis = () => {
    setResult(null)
    setUploadedImage(null)
    setError(null)
  }

  const handleLogout = async () => {
    await supabase.auth.signOut()
    router.push("/")
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <Brain className="h-12 w-12 text-primary/50 mx-auto mb-4 animate-pulse" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card sticky top-0 z-40">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <Link href="/dashboard" className="flex items-center space-x-3">
              <div className="w-8 h-8 sm:w-10 sm:h-10 bg-primary rounded-lg flex items-center justify-center">
                <Microscope className="h-5 w-5 sm:h-6 sm:w-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="font-serif text-xl sm:text-2xl font-bold text-foreground">HistoAI</h1>
                <p className="font-sans text-xs sm:text-sm text-muted-foreground">Analysis</p>
              </div>
            </Link>

            <div className="flex items-center gap-2 sm:gap-4">
              <Link href="/dashboard">
                <Button variant="ghost" size="sm" className="text-xs sm:text-sm">
                  <BarChart3 className="h-4 w-4 mr-2" />
                  <span className="hidden sm:inline">Dashboard</span>
                </Button>
              </Link>
              <Button variant="ghost" size="sm" onClick={handleLogout} className="text-xs sm:text-sm">
                <LogOut className="h-4 w-4 mr-2" />
                <span className="hidden sm:inline">Logout</span>
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">
        {error && (
          <Alert variant="destructive" className="mb-4 sm:mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-sm whitespace-pre-line">{error}</AlertDescription>
          </Alert>
        )}

        {apiStatus === "offline" && (
          <Alert className="mb-4 sm:mb-6 border-orange-200 bg-orange-50 text-orange-800">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-sm">
              API server is offline. Please start your FastAPI server and ensure it's running at{" "}
              <code className="bg-orange-100 px-1 rounded text-xs">
                {process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}
              </code>
            </AlertDescription>
          </Alert>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="space-y-4">
            <h2 className="text-2xl font-bold text-foreground">Upload Image</h2>
            <ImageUpload onImageUpload={handleImageUpload} />
            {uploadedImage && (
              <div className="mt-4">
                <img
                  src={uploadedImage || "/placeholder.svg"}
                  alt="Uploaded"
                  className="w-full h-auto rounded-lg border border-border"
                />
              </div>
            )}
          </div>

          <div className="space-y-4">
            <h2 className="text-2xl font-bold text-foreground">Analysis Results</h2>
            {result && <ResultsDisplay result={result} />}
            {isAnalyzing && (
              <div className="text-center py-4">
                <div className="inline-block animate-spin">
                  <RefreshCw className="h-8 w-8 text-primary/50" />
                </div>
                <p className="text-muted-foreground mt-2">Analyzing image...</p>
              </div>
            )}
          </div>
        </div>

        {result && (
          <div className="mt-8">
            <h3 className="text-xl font-semibold text-foreground mb-4">Prediction Details</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Prediction</label>
                <p className="text-foreground font-medium">{result.prediction}</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Confidence</label>
                <p className="text-foreground font-medium">{result.confidence.toFixed(2)}%</p>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Processing Time</label>
                <p className="text-foreground font-medium">
                  {result.processing_time ? `${result.processing_time.toFixed(2)}s` : "N/A"}
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="mt-8">
          <Button onClick={resetAnalysis} variant="outline" className="w-full sm:w-auto bg-transparent">
            <RefreshCw className="h-4 w-4 mr-2" />
            Reset Analysis
          </Button>
        </div>
      </main>
    </div>
  )
}
