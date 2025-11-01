"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Microscope, Brain, Eye, Shield, ArrowRight, Zap, BarChart3 } from "lucide-react"

export default function LandingPage() {
  const [isScrolled, setIsScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50)
    }
    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav
        className={`fixed top-0 w-full z-50 transition-all duration-300 ${
          isScrolled ? "bg-card border-b border-border shadow-sm" : "bg-transparent"
        }`}
      >
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 sm:w-10 sm:h-10 bg-primary rounded-lg flex items-center justify-center">
              <Microscope className="h-5 w-5 sm:h-6 sm:w-6 text-primary-foreground" />
            </div>
            <h1 className="font-serif text-xl sm:text-2xl font-bold text-foreground">HistoAI</h1>
          </div>

          <div className="flex items-center gap-2 sm:gap-4">
            <Link href="/auth/login">
              <Button variant="ghost" className="text-xs sm:text-sm">
                Login
              </Button>
            </Link>
            <Link href="/auth/sign-up">
              <Button className="text-xs sm:text-sm">Get Started</Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 sm:pt-40 pb-16 sm:pb-24 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
        <div className="absolute inset-0 -z-10 opacity-30">
          <div className="absolute top-20 left-10 w-72 h-72 bg-primary/20 rounded-full blur-3xl"></div>
          <div className="absolute bottom-20 right-10 w-72 h-72 bg-accent/20 rounded-full blur-3xl"></div>
        </div>

        <div className="container mx-auto max-w-4xl text-center space-y-6 sm:space-y-8">
          <div className="inline-block px-3 sm:px-4 py-1 sm:py-2 bg-primary/10 rounded-full border border-primary/20">
            <p className="text-xs sm:text-sm font-medium text-primary">AI-Powered Medical Analysis</p>
          </div>

          <h1 className="font-serif text-4xl sm:text-5xl lg:text-6xl font-bold text-foreground leading-tight">
            Advanced Histopathology Cancer Detection
          </h1>

          <p className="font-sans text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto">
            Harness the power of artificial intelligence to analyze histopathology images with unprecedented accuracy.
            Get instant insights with explainable AI visualizations.
          </p>

          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center pt-4 sm:pt-6">
            <Link href="/auth/sign-up">
              <Button size="lg" className="w-full sm:w-auto text-sm sm:text-base">
                Start Analysis
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
            <Link href="#features">
              <Button variant="outline" size="lg" className="w-full sm:w-auto text-sm sm:text-base bg-transparent">
                Learn More
              </Button>
            </Link>
          </div>
        </div>

        {/* Hero Image */}
        <div className="mt-12 sm:mt-16 container mx-auto max-w-3xl">
          <div className="relative rounded-2xl overflow-hidden border border-border bg-card p-4 sm:p-8 shadow-lg">
            <img src="/medical-histopathology-microscope-analysis-interfa.jpg" alt="HistoAI Interface" className="w-full h-auto rounded-lg" />
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-16 sm:py-24 px-4 sm:px-6 lg:px-8 bg-card/50">
        <div className="container mx-auto">
          <div className="text-center space-y-3 sm:space-y-4 mb-12 sm:mb-16">
            <h2 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-bold text-foreground">Powerful Features</h2>
            <p className="font-sans text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto">
              Everything you need for professional histopathology analysis
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
            {/* Feature 1 */}
            <Card className="border-border hover:shadow-lg transition-shadow duration-300">
              <CardContent className="p-6 sm:p-8">
                <div className="w-12 h-12 sm:w-14 sm:h-14 bg-primary/10 rounded-lg flex items-center justify-center mb-4 sm:mb-6">
                  <Brain className="h-6 w-6 sm:h-7 sm:w-7 text-primary" />
                </div>
                <h3 className="font-serif text-lg sm:text-xl font-semibold mb-2 sm:mb-3">AI-Powered Analysis</h3>
                <p className="font-sans text-sm sm:text-base text-muted-foreground">
                  Advanced deep learning models trained on extensive histopathology datasets for accurate predictions
                </p>
              </CardContent>
            </Card>

            {/* Feature 2 */}
            <Card className="border-border hover:shadow-lg transition-shadow duration-300">
              <CardContent className="p-6 sm:p-8">
                <div className="w-12 h-12 sm:w-14 sm:h-14 bg-primary/10 rounded-lg flex items-center justify-center mb-4 sm:mb-6">
                  <Eye className="h-6 w-6 sm:h-7 sm:w-7 text-primary" />
                </div>
                <h3 className="font-serif text-lg sm:text-xl font-semibold mb-2 sm:mb-3">Explainable Results</h3>
                <p className="font-sans text-sm sm:text-base text-muted-foreground">
                  Grad-CAM heatmaps show exactly where the AI focused during analysis for transparency
                </p>
              </CardContent>
            </Card>

            {/* Feature 3 */}
            <Card className="border-border hover:shadow-lg transition-shadow duration-300 sm:col-span-2 lg:col-span-1">
              <CardContent className="p-6 sm:p-8">
                <div className="w-12 h-12 sm:w-14 sm:h-14 bg-primary/10 rounded-lg flex items-center justify-center mb-4 sm:mb-6">
                  <BarChart3 className="h-6 w-6 sm:h-7 sm:w-7 text-primary" />
                </div>
                <h3 className="font-serif text-lg sm:text-xl font-semibold mb-2 sm:mb-3">Analysis History</h3>
                <p className="font-sans text-sm sm:text-base text-muted-foreground">
                  Keep track of all your analyses with detailed history and insights
                </p>
              </CardContent>
            </Card>

            {/* Feature 4 */}
            <Card className="border-border hover:shadow-lg transition-shadow duration-300">
              <CardContent className="p-6 sm:p-8">
                <div className="w-12 h-12 sm:w-14 sm:h-14 bg-primary/10 rounded-lg flex items-center justify-center mb-4 sm:mb-6">
                  <Zap className="h-6 w-6 sm:h-7 sm:w-7 text-primary" />
                </div>
                <h3 className="font-serif text-lg sm:text-xl font-semibold mb-2 sm:mb-3">Fast Processing</h3>
                <p className="font-sans text-sm sm:text-base text-muted-foreground">
                  Get results in seconds with our optimized AI infrastructure
                </p>
              </CardContent>
            </Card>

            {/* Feature 5 */}
            <Card className="border-border hover:shadow-lg transition-shadow duration-300">
              <CardContent className="p-6 sm:p-8">
                <div className="w-12 h-12 sm:w-14 sm:h-14 bg-primary/10 rounded-lg flex items-center justify-center mb-4 sm:mb-6">
                  <Shield className="h-6 w-6 sm:h-7 sm:w-7 text-primary" />
                </div>
                <h3 className="font-serif text-lg sm:text-xl font-semibold mb-2 sm:mb-3">Research Grade</h3>
                <p className="font-sans text-sm sm:text-base text-muted-foreground">
                  Built for educational and research purposes with clinical-grade accuracy
                </p>
              </CardContent>
            </Card>

            {/* Feature 6 */}
            <Card className="border-border hover:shadow-lg transition-shadow duration-300 sm:col-span-2 lg:col-span-1">
              <CardContent className="p-6 sm:p-8">
                <div className="w-12 h-12 sm:w-14 sm:h-14 bg-primary/10 rounded-lg flex items-center justify-center mb-4 sm:mb-6">
                  <Microscope className="h-6 w-6 sm:h-7 sm:w-7 text-primary" />
                </div>
                <h3 className="font-serif text-lg sm:text-xl font-semibold mb-2 sm:mb-3">Professional Tools</h3>
                <p className="font-sans text-sm sm:text-base text-muted-foreground">
                  Comprehensive dashboard with advanced analysis and reporting capabilities
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-16 sm:py-24 px-4 sm:px-6 lg:px-8">
        <div className="container mx-auto max-w-4xl">
          <div className="text-center space-y-3 sm:space-y-4 mb-12 sm:mb-16">
            <h2 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-bold text-foreground">How It Works</h2>
            <p className="font-sans text-base sm:text-lg text-muted-foreground">
              Simple three-step process for professional analysis
            </p>
          </div>

          <div className="space-y-6 sm:space-y-8">
            {/* Step 1 */}
            <div className="flex gap-4 sm:gap-6">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 sm:h-14 sm:w-14 rounded-lg bg-primary text-primary-foreground font-serif font-bold text-lg sm:text-xl">
                  1
                </div>
              </div>
              <div className="flex-1">
                <h3 className="font-serif text-lg sm:text-xl font-semibold mb-2">Upload Your Image</h3>
                <p className="font-sans text-sm sm:text-base text-muted-foreground">
                  Simply upload your histopathology image in JPG or PNG format. Our system supports high-resolution
                  images for detailed analysis.
                </p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="flex gap-4 sm:gap-6">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 sm:h-14 sm:w-14 rounded-lg bg-primary text-primary-foreground font-serif font-bold text-lg sm:text-xl">
                  2
                </div>
              </div>
              <div className="flex-1">
                <h3 className="font-serif text-lg sm:text-xl font-semibold mb-2">AI Analysis</h3>
                <p className="font-sans text-sm sm:text-base text-muted-foreground">
                  Our advanced AI model processes your image and generates predictions with confidence scores and
                  detailed analysis.
                </p>
              </div>
            </div>

            {/* Step 3 */}
            <div className="flex gap-4 sm:gap-6">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 sm:h-14 sm:w-14 rounded-lg bg-primary text-primary-foreground font-serif font-bold text-lg sm:text-xl">
                  3
                </div>
              </div>
              <div className="flex-1">
                <h3 className="font-serif text-lg sm:text-xl font-semibold mb-2">Get Results</h3>
                <p className="font-sans text-sm sm:text-base text-muted-foreground">
                  Receive comprehensive results with Grad-CAM heatmaps showing AI attention areas and detailed
                  explanations.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 sm:py-24 px-4 sm:px-6 lg:px-8 bg-primary text-primary-foreground">
        <div className="container mx-auto max-w-3xl text-center space-y-6 sm:space-y-8">
          <h2 className="font-serif text-3xl sm:text-4xl lg:text-5xl font-bold">Ready to Get Started?</h2>
          <p className="font-sans text-base sm:text-lg opacity-90">
            Join researchers and professionals using HistoAI for advanced histopathology analysis
          </p>
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center pt-4 sm:pt-6">
            <Link href="/auth/sign-up">
              <Button size="lg" variant="secondary" className="w-full sm:w-auto text-sm sm:text-base">
                Create Account
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </Link>
            <Link href="/auth/login">
              <Button
                size="lg"
                variant="outline"
                className="w-full sm:w-auto text-sm sm:text-base bg-transparent border-primary-foreground text-primary-foreground hover:bg-primary-foreground/10"
              >
                Sign In
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border bg-card py-8 sm:py-12 px-4 sm:px-6 lg:px-8">
        <div className="container mx-auto">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8 mb-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                  <Microscope className="h-5 w-5 text-primary-foreground" />
                </div>
                <h3 className="font-serif font-bold text-foreground">HistoAI</h3>
              </div>
              <p className="font-sans text-sm text-muted-foreground">
                Advanced AI-powered histopathology analysis for research and education.
              </p>
            </div>

            <div>
              <h4 className="font-serif font-semibold text-foreground mb-4">Product</h4>
              <ul className="space-y-2 font-sans text-sm text-muted-foreground">
                <li>
                  <a href="#features" className="hover:text-foreground transition-colors">
                    Features
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-foreground transition-colors">
                    Pricing
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-foreground transition-colors">
                    Documentation
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h4 className="font-serif font-semibold text-foreground mb-4">Company</h4>
              <ul className="space-y-2 font-sans text-sm text-muted-foreground">
                <li>
                  <a href="#" className="hover:text-foreground transition-colors">
                    About
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-foreground transition-colors">
                    Blog
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-foreground transition-colors">
                    Contact
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h4 className="font-serif font-semibold text-foreground mb-4">Legal</h4>
              <ul className="space-y-2 font-sans text-sm text-muted-foreground">
                <li>
                  <a href="#" className="hover:text-foreground transition-colors">
                    Privacy
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-foreground transition-colors">
                    Terms
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-foreground transition-colors">
                    Disclaimer
                  </a>
                </li>
              </ul>
            </div>
          </div>

          <div className="border-t border-border pt-8">
            <p className="font-sans text-xs sm:text-sm text-muted-foreground text-center">
              © 2025 HistoAI. Built for educational and research purposes. Always consult medical professionals for
              clinical decisions.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}
