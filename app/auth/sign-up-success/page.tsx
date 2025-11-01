"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import Link from "next/link"
import { Microscope, CheckCircle, Mail } from "lucide-react"

export default function SignUpSuccessPage() {
  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <Link href="/" className="flex items-center space-x-3 w-fit">
            <div className="w-8 h-8 sm:w-10 sm:h-10 bg-primary rounded-lg flex items-center justify-center">
              <Microscope className="h-5 w-5 sm:h-6 sm:w-6 text-primary-foreground" />
            </div>
            <h1 className="font-serif text-xl sm:text-2xl font-bold text-foreground">HistoAI</h1>
          </Link>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        <Card className="w-full max-w-md">
          <CardContent className="p-6 sm:p-8 text-center space-y-6">
            <div className="flex justify-center">
              <div className="w-16 h-16 sm:w-20 sm:h-20 bg-green-100 rounded-full flex items-center justify-center">
                <CheckCircle className="h-8 w-8 sm:h-10 sm:w-10 text-green-600" />
              </div>
            </div>

            <div className="space-y-2">
              <h1 className="font-serif text-2xl sm:text-3xl font-bold text-foreground">Account Created!</h1>
              <p className="font-sans text-muted-foreground">Your account has been successfully created.</p>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-2">
              <div className="flex items-start gap-3">
                <Mail className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                <div className="text-left">
                  <p className="font-medium text-blue-900 text-sm">Verify Your Email</p>
                  <p className="text-xs text-blue-700 mt-1">
                    We've sent a confirmation email. Please check your inbox and click the link to verify your account.
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <Link href="/auth/login" className="block">
                <Button className="w-full">Go to Login</Button>
              </Link>
              <Link href="/" className="block">
                <Button variant="outline" className="w-full bg-transparent">
                  Back to Home
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
