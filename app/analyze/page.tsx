"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { createClient } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Microscope, AlertCircle, RefreshCw, LogOut, BarChart3, Image as ImageIcon, Sparkles } from "lucide-react"
import { unifiedAPI } from "@/services/unified-api"
import type { UnifiedPredictionResult } from "@/services/unified-api"
import Link from "next/link"
import { SimpleImageUpload } from "@/components/simple-image-upload"
import { ModernPredictionResults } from "@/components/modern-prediction-results"

export default function AnalyzePage() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<UnifiedPredictionResult | null>(null)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [heatmapImage, setHeatmapImage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [user, setUser] = useState<any>(null)
  const router = useRouter()
  const supabase = createClient()

  useEffect(() => {
    const checkAuth = async () => {
      const { data: { user: currentUser } } = await supabase.auth.getUser()
      if (!currentUser) {
        router.push("/auth/login")
        return
      }
      setUser(currentUser)
      setIsLoading(false)
    }
    checkAuth()
  }, [])

  const handleImageUpload = (file: File) => {
    setIsAnalyzing(true);
    setResult(null);
    setError(null);
    setHeatmapImage(null);

    const reader = new FileReader();
    reader.onload = async (e) => {
      const imageUrl = e.target?.result as string;
      setUploadedImage(imageUrl);

      try {
        const prediction = await unifiedAPI.predictCancer(file, 'backend');
        setResult(prediction);
        if (prediction.heatmap) {
          setHeatmapImage(prediction.heatmap);
        }

        if (user && imageUrl) {
          const { error: dbError } = await supabase.from("analysis_history").insert({
            user_id: user.id,
            prediction: prediction.prediction,
            confidence: prediction.confidence,
            image_url: imageUrl,
            probabilities: prediction.probabilities,
            heatmap: prediction.heatmap, // Save the heatmap
          });
          if (dbError) {
            throw new Error(`Database error: ${dbError.message}`);
          }
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "An unexpected error occurred.";
        setError(errorMessage);
      } finally {
        setIsAnalyzing(false);
      }
    };
    reader.onerror = () => {
      setError("Failed to read the image file.");
      setIsAnalyzing(false);
    };
    reader.readAsDataURL(file);
  };

  const resetAnalysis = () => {
    setResult(null)
    setUploadedImage(null)
    setError(null)
    setHeatmapImage(null)
  }

  const handleLogout = async () => {
    await supabase.auth.signOut()
    router.push("/")
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <RefreshCw className="h-12 w-12 text-gray-400 animate-spin" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800">
      <header className="bg-white/80 backdrop-blur-lg border-b border-gray-200 sticky top-0 z-40">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link href="/dashboard" className="flex items-center gap-2">
              <Microscope className="h-7 w-7 text-green-600" />
              <h1 className="text-xl font-bold text-gray-800">HistoAI</h1>
            </Link>
            <div className="flex items-center gap-4">
              <Link href="/dashboard">
                <Button variant="ghost" size="sm"><BarChart3 className="h-4 w-4 mr-2" />Dashboard</Button>
              </Link>
              <Button variant="ghost" size="sm" onClick={handleLogout}><LogOut className="h-4 w-4 mr-2" />Logout</Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {!result && (
          <div className="max-w-3xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">AI-Powered Cancer Detection</h2>
              <p className="mt-4 text-lg text-gray-600">Upload a histopathology image to get an instant analysis and heatmap visualization.</p>
            </div>
            <SimpleImageUpload onImageUpload={handleImageUpload} isLoading={isAnalyzing} />
          </div>
        )}

        {result && (
          <div>
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-3xl font-bold tracking-tight text-gray-900">Analysis Complete</h2>
              <Button onClick={resetAnalysis} variant="outline"><RefreshCw className="h-4 w-4 mr-2" />New Analysis</Button>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
              <div className="lg:col-span-1 space-y-4">
                <div className="p-2 bg-white rounded-lg border">
                  <h3 className="text-sm font-semibold text-gray-500 mb-2 px-2">ORIGINAL IMAGE</h3>
                  {uploadedImage && <img src={uploadedImage} alt="Uploaded" className="rounded-md w-full" />}
                </div>
              </div>

              <div className="lg:col-span-1">
                <ModernPredictionResults results={result} />
              </div>

              <div className="lg:col-span-1 space-y-4">
                <div className="p-2 bg-white rounded-lg border">
                  <h3 className="text-sm font-semibold text-gray-500 mb-2 px-2">GRAD-CAM HEATMAP</h3>
                  {heatmapImage && <img src={heatmapImage} alt="Grad-CAM Heatmap" className="rounded-md w-full" />}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
