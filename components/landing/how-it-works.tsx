"use client"

import { ArrowRight } from "lucide-react"

const steps = [
  {
    number: "01",
    title: "Upload Image",
    description: "Select or drag your histopathology image to begin analysis",
  },
  {
    number: "02",
    title: "AI Processing",
    description: "Our advanced model analyzes the image in real-time",
  },
  {
    number: "03",
    title: "Get Results",
    description: "Receive detailed predictions with confidence scores",
  },
  {
    number: "04",
    title: "View Insights",
    description: "Explore Grad-CAM heatmaps for explainable AI",
  },
]

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="py-20 sm:py-32 px-4 sm:px-6 lg:px-8 bg-slate-50">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16 sm:mb-20">
          <h2 className="text-4xl sm:text-5xl font-bold text-slate-900 mb-4">How It Works</h2>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Simple, intuitive process from image upload to actionable insights
          </p>
        </div>

        {/* Steps */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {steps.map((step, index) => (
            <div key={index} className="relative">
              {/* Step card */}
              <div className="bg-white rounded-xl p-8 border border-slate-200 hover:border-green-300 transition-colors duration-300 h-full">
                <div className="text-5xl font-bold text-green-600/20 mb-4">{step.number}</div>
                <h3 className="text-xl font-semibold text-slate-900 mb-2">{step.title}</h3>
                <p className="text-slate-600">{step.description}</p>
              </div>

              {/* Arrow connector */}
              {index < steps.length - 1 && (
                <div className="hidden lg:flex absolute -right-4 top-1/2 -translate-y-1/2 z-10">
                  <ArrowRight className="w-8 h-8 text-green-600/30" />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
