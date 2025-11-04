"use client"

import { useEffect, useRef } from "react"
import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function HeroSection() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return

      const rect = containerRef.current.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top

      const moveX = (x - rect.width / 2) * 0.02
      const moveY = (y - rect.height / 2) * 0.02

      const elements = containerRef.current.querySelectorAll("[data-parallax]")
      elements.forEach((el) => {
        const element = el as HTMLElement
        const speed = Number.parseFloat(element.getAttribute("data-parallax") || "1")
        element.style.transform = `translate(${moveX * speed}px, ${moveY * speed}px)`
      })
    }

    window.addEventListener("mousemove", handleMouseMove)
    return () => window.removeEventListener("mousemove", handleMouseMove)
  }, [])

  return (
    <section
      ref={containerRef}
      className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20"
    >
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div data-parallax="0.5" className="absolute top-20 right-10 w-72 h-72 bg-green-200/20 rounded-full blur-3xl" />
        <div
          data-parallax="-0.3"
          className="absolute bottom-20 left-10 w-96 h-96 bg-lime-200/20 rounded-full blur-3xl"
        />
      </div>

      <div className="relative z-10 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-50 border border-green-200 rounded-full mb-8 animate-fade-in">
          <span className="w-2 h-2 bg-green-600 rounded-full animate-pulse" />
          <span className="text-sm font-medium text-green-700">AI-Powered Medical Analysis</span>
        </div>

        {/* Main heading */}
        <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-slate-900 mb-6 leading-tight animate-fade-in-up">
          Intelligent Histopathology
          <span className="block bg-gradient-to-r from-green-600 to-lime-500 bg-clip-text text-transparent">
            Cancer Detection
          </span>
        </h1>

        {/* Subheading */}
        <p className="text-lg sm:text-xl text-slate-600 mb-8 max-w-2xl mx-auto leading-relaxed animate-fade-in-up animation-delay-100">
          Harness the power of advanced AI to analyze histopathology images with unprecedented accuracy. Get instant
          predictions and explainable insights for better clinical decision-making.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16 animate-fade-in-up animation-delay-200">
          <Link href="/analyze">
            <Button className="bg-green-600 hover:bg-green-700 text-white px-8 py-6 text-lg rounded-full group">
              Start Analysis
              <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Button>
          </Link>
          <Button
            variant="outline"
            className="px-8 py-6 text-lg rounded-full border-slate-300 hover:bg-slate-50 bg-transparent"
          >
            Watch Demo
          </Button>
        </div>

        {/* Hero image placeholder with animation */}
        <div className="relative animate-fade-in-up animation-delay-300">
          <div className="absolute inset-0 bg-gradient-to-r from-green-600/20 to-lime-500/20 rounded-2xl blur-2xl" />
          <div className="relative bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl p-8 border border-slate-700/50 overflow-hidden">
            <div className="absolute inset-0 bg-grid-pattern opacity-10" />
            <svg
              className="w-full h-auto text-slate-700"
              viewBox="0 0 600 400"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <rect x="50" y="50" width="500" height="300" rx="12" fill="currentColor" opacity="0.1" />
              <circle cx="300" cy="200" r="80" fill="none" stroke="currentColor" strokeWidth="2" opacity="0.3" />
              <circle cx="300" cy="200" r="60" fill="none" stroke="currentColor" strokeWidth="2" opacity="0.5" />
              <circle cx="300" cy="200" r="40" fill="none" stroke="currentColor" strokeWidth="2" opacity="0.7" />
              <path d="M 300 160 Q 340 200 300 240 Q 260 200 300 160" fill="currentColor" opacity="0.3" />
            </svg>
          </div>
        </div>
      </div>
    </section>
  )
}
