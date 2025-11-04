"use client"

import { useState, useEffect } from "react"
import HeroSection from "@/components/landing/hero-section"
import FeaturesSection from "@/components/landing/features-section"
import HowItWorks from "@/components/landing/how-it-works"
import CTASection from "@/components/landing/cta-section"
import Navigation from "@/components/landing/navigation"

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
    <div className="min-h-screen bg-gradient-to-b from-background via-background to-slate-50">
      <Navigation isScrolled={isScrolled} />
      <HeroSection />
      <FeaturesSection />
      <HowItWorks />
      <CTASection />
    </div>
  )
}
