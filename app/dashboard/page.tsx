"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { createClient } from "@/lib/supabase/client"
import { Button } from "@/components/ui/button"
import { StatCard } from "@/components/StatCard"
import { HistoryCard } from "@/components/HistoryCard"
import { HistoryDetailModal } from "@/components/HistoryDetailModal"
import { Microscope, Activity, AlertTriangle, Percent, LogOut, Plus, Bot } from "lucide-react"
import { GeminiChat } from "@/components/GeminiChat"
import Link from "next/link"

export default function Dashboard() {
  const [user, setUser] = useState<any>(null)
  const [history, setHistory] = useState<any[]>([])
  const [stats, setStats] = useState({ total: 0, cancerous: 0, avgConfidence: 0 })
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()
  const supabase = createClient()

  useEffect(() => {
    const fetchData = async () => {
      const { data: { user: currentUser } } = await supabase.auth.getUser()
      if (!currentUser) {
        router.push("/auth/login")
        return
      }
      setUser(currentUser)

      const { data: historyData, error } = await supabase
        .from("analysis_history")
        .select("*")
        .eq("user_id", currentUser.id)
        .order("created_at", { ascending: false })

      if (historyData) {
        setHistory(historyData)
        calculateStats(historyData)
      }
      setIsLoading(false)
    }

    fetchData()
  }, [])

  const calculateStats = (data: any[]) => {
    const total = data.length
    const cancerous = data.filter(item => item.prediction === 'malignant').length
    const avgConfidence = total > 0 ? data.reduce((acc, item) => acc + item.confidence, 0) / total : 0
    setStats({ total, cancerous, avgConfidence })
  }

  const handleLogout = async () => {
    await supabase.auth.signOut()
    router.push("/")
  }

  if (isLoading) {
    return <div className="min-h-screen bg-gray-50 flex items-center justify-center">Loading...</div>
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
              <Link href="/analyze">
                <Button><Plus className="h-4 w-4 mr-2" />New Analysis</Button>
              </Link>
              <Button variant="ghost" size="sm" onClick={handleLogout}><LogOut className="h-4 w-4 mr-2" />Logout</Button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="rounded-lg bg-gradient-to-r from-green-100 to-green-200 p-8 mb-8">
            <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">Welcome back, {user?.user_metadata?.full_name || user?.email}!</h2>
            <p className="mt-4 text-lg text-gray-700">Get started by uploading a new image for analysis or review your past results.</p>
            <div className="mt-6">
                <Link href="/analyze">
                    <Button size="lg" className="bg-green-600 hover:bg-green-700 text-white">Start New Analysis</Button>
                </Link>
            </div>
        </div>

        <div className="grid gap-6 md:grid-cols-3 mb-8">
          <StatCard title="Total Analyses" value={stats.total} icon={Activity} />
          <StatCard title="Cancerous Cases" value={stats.cancerous} icon={AlertTriangle} />
          <StatCard title="Avg. Confidence" value={`${(stats.avgConfidence * 100).toFixed(1)}%`} icon={Percent} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <h3 className="text-2xl font-bold tracking-tight text-gray-900 mb-4">Analysis History</h3>
            {history.length === 0 ? (
              <div className="text-center py-16 border-2 border-dashed border-gray-300 rounded-lg">
                <Microscope className="h-12 w-12 mx-auto text-gray-400 mb-4" />
                <h4 className="text-lg font-medium text-gray-800">No analyses yet</h4>
                <p className="text-gray-500 mt-2 mb-6">Your analysis history will appear here.</p>
                <Link href="/analyze"><Button>Start First Analysis</Button></Link>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {history.map(item => (
                  <HistoryDetailModal key={item.id} analysis={item}>
                    <div className="cursor-pointer">
                      <HistoryCard analysis={item} />
                    </div>
                  </HistoryDetailModal>
                ))}
              </div>
            )}
          </div>

          <div className="lg:col-span-1">
            <h3 className="text-2xl font-bold tracking-tight text-gray-900 mb-4">Chat with Gemini</h3>
            <div className="bg-white p-6 rounded-lg border border-gray-200">
              <div className="flex items-center gap-3 mb-4">
                <Bot className="h-6 w-6 text-green-600" />
                <h4 className="text-lg font-semibold">Gemini Assistant</h4>
              </div>
              <GeminiChat />
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
