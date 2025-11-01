"use client"

import { useEffect, useState } from "react"
import { createClient } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Microscope, LogOut, Plus, Trash2, Calendar, TrendingUp } from "lucide-react"
import { GeminiChat } from "@/components/gemini-chat"
import { histopathologyAPI as api } from "@/services/api"

interface Analysis {
  id: string
  prediction: string
  confidence: number
  created_at: string
  image_url: string
}

export default function DashboardPage() {
  const [analyses, setAnalyses] = useState<Analysis[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [user, setUser] = useState<any>(null)
  const [isChatLoading, setIsChatLoading] = useState(false) // New state for chat loading
  const router = useRouter()
  const supabase = createClient()

  useEffect(() => {
    const loadData = async () => {
      try {
        // Get current user
        const {
          data: { user: currentUser },
        } = await supabase.auth.getUser()

        if (!currentUser) {
          router.push("/auth/login")
          return
        }

        setUser(currentUser)

        // Fetch analysis history
        const { data, error: fetchError } = await supabase
          .from("analysis_history")
          .select("*")
          .eq("user_id", currentUser.id)
          .order("created_at", { ascending: false })

        if (fetchError) throw fetchError
        setAnalyses(data || [])
      } catch (err) {
        console.error("Error loading data:", err)
        setError("Failed to load your analysis history")
      } finally {
        setIsLoading(false)
      }
    }

    loadData()
  }, [])

  const handleLogout = async () => {
    await supabase.auth.signOut()
    router.push("/")
  }

  const handleDelete = async (id: string) => {
    try {
      const { error } = await supabase.from("analysis_history").delete().eq("id", id)

      if (error) throw error
      setAnalyses(analyses.filter((a) => a.id !== id))
    } catch (err) {
      console.error("Error deleting analysis:", err)
      setError("Failed to delete analysis")
    }
  }

  const handleSendMessage = async (message: string) => {
    setIsChatLoading(true)
    try {
      const response = await api.chatWithGemini(message)
      return response
    } catch (error) {
      console.error("Error sending chat message:", error)
      return "Error: Could not get a response from Gemini."
    } finally {
      setIsChatLoading(false)
    }
  }

  const cancerousCount = analyses.filter((a) => a.prediction === "Cancerous").length
  const averageConfidence =
    analyses.length > 0 ? ((analyses.reduce((sum, a) => sum + a.confidence, 0) / analyses.length) * 100).toFixed(1) : 0

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card sticky top-0 z-40">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <Link href="/" className="flex items-center space-x-3">
              <div className="w-8 h-8 sm:w-10 sm:h-10 bg-primary rounded-lg flex items-center justify-center">
                <Microscope className="h-5 w-5 sm:h-6 sm:w-6 text-primary-foreground" />
              </div>
              <h1 className="font-serif text-xl sm:text-2xl font-bold text-foreground">HistoAI</h1>
            </Link>

            <div className="flex items-center gap-2 sm:gap-4">
              <Link href="/analyze">
                <Button size="sm" className="text-xs sm:text-sm">
                  <Plus className="h-4 w-4 mr-2" />
                  New Analysis
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

      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        {/* Welcome Section */}
        <div className="mb-8 sm:mb-12">
          <h2 className="font-serif text-2xl sm:text-3xl font-bold text-foreground mb-2">
            Welcome back, {user?.email?.split("@")[0]}!
          </h2>
          <p className="font-sans text-muted-foreground">Manage and review your analysis history</p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-6 mb-8 sm:mb-12">
          <Card>
            <CardContent className="p-4 sm:p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-sans text-xs sm:text-sm text-muted-foreground">Total Analyses</p>
                  <p className="font-serif text-2xl sm:text-3xl font-bold text-foreground mt-1">{analyses.length}</p>
                </div>
                <TrendingUp className="h-8 w-8 sm:h-10 sm:w-10 text-primary/20" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4 sm:p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-sans text-xs sm:text-sm text-muted-foreground">Cancerous Cases</p>
                  <p className="font-serif text-2xl sm:text-3xl font-bold text-red-600 mt-1">{cancerousCount}</p>
                </div>
                <div className="h-8 w-8 sm:h-10 sm:w-10 bg-red-100 rounded-lg flex items-center justify-center">
                  <span className="text-red-600 font-bold text-sm">!</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4 sm:p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-sans text-xs sm:text-sm text-muted-foreground">Avg. Confidence</p>
                  <p className="font-serif text-2xl sm:text-3xl font-bold text-primary mt-1">{averageConfidence}%</p>
                </div>
                <div className="h-8 w-8 sm:h-10 sm:w-10 bg-primary/10 rounded-lg flex items-center justify-center">
                  <span className="text-primary font-bold text-sm">%</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Analysis History */}
        <div>
          <h3 className="font-serif text-xl sm:text-2xl font-bold text-foreground mb-4 sm:mb-6">Analysis History</h3>

          {error && (
            <Alert variant="destructive" className="mb-4 sm:mb-6">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {isLoading ? (
            <Card>
              <CardContent className="p-8 text-center">
                <p className="text-muted-foreground">Loading your analyses...</p>
              </CardContent>
            </Card>
          ) : analyses.length === 0 ? (
            <Card>
              <CardContent className="p-8 sm:p-12 text-center">
                <Microscope className="h-12 w-12 sm:h-16 sm:w-16 text-muted-foreground/30 mx-auto mb-4" />
                <h3 className="font-serif text-lg sm:text-xl font-semibold text-foreground mb-2">No analyses yet</h3>
                <p className="font-sans text-sm text-muted-foreground mb-6">
                  Start by uploading your first histopathology image for analysis
                </p>
                <Link href="/analyze">
                  <Button>
                    <Plus className="h-4 w-4 mr-2" />
                    Start Analysis
                  </Button>
                </Link>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
              {analyses.map((analysis) => (
                <Card key={analysis.id} className="overflow-hidden hover:shadow-lg transition-shadow">
                  <div className="aspect-square bg-muted overflow-hidden">
                    <img
                      src={analysis.image_url || "/placeholder.svg"}
                      alt="Analysis"
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        ;(e.target as HTMLImageElement).src = "/histopathology-image.jpg"
                      }}
                    />
                  </div>
                  <CardContent className="p-4 sm:p-6">
                    <div className="space-y-3 sm:space-y-4">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-sans text-xs sm:text-sm text-muted-foreground">Prediction</span>
                          <span
                            className={`font-serif font-bold text-sm sm:text-base ${
                              analysis.prediction === "Cancerous" ? "text-red-600" : "text-green-600"
                            }`}
                          >
                            {analysis.prediction}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="font-sans text-xs sm:text-sm text-muted-foreground">Confidence</span>
                          <span className="font-serif font-bold text-sm sm:text-base text-primary">
                            {(analysis.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center gap-2 text-xs sm:text-sm text-muted-foreground">
                        <Calendar className="h-3 w-3 sm:h-4 sm:w-4" />
                        {new Date(analysis.created_at).toLocaleDateString()}
                      </div>

                      <div className="flex gap-2 pt-2 sm:pt-4">
                        <Link href={`/analyze?id=${analysis.id}`} className="flex-1">
                          <Button variant="outline" size="sm" className="w-full text-xs sm:text-sm bg-transparent">
                            View
                          </Button>
                        </Link>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDelete(analysis.id)}
                          className="text-xs sm:text-sm"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>

        {/* Gemini Chat */}
        <Card className="mt-8 sm:mt-12">
          <CardContent className="p-4 sm:p-6">
            <h3 className="font-serif text-xl sm:text-2xl font-bold text-foreground mb-4 sm:mb-6">Chat with Gemini</h3>
            <GeminiChat onSendMessage={handleSendMessage} isLoading={isChatLoading} />
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
