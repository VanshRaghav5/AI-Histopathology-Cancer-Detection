"use client"

import { Upload, Brain, Zap, Shield, BarChart3, Lock } from "lucide-react"

const features = [
  {
    icon: Upload,
    title: "Easy Image Upload",
    description: "Drag and drop histopathology images for instant analysis. Supports JPG and PNG formats.",
    color: "from-blue-500 to-blue-600",
  },
  {
    icon: Brain,
    title: "Advanced AI Model",
    description: "Powered by state-of-the-art deep learning models trained on extensive medical datasets.",
    color: "from-purple-500 to-purple-600",
  },
  {
    icon: Zap,
    title: "Lightning Fast",
    description: "Get analysis results in seconds with optimized processing pipelines.",
    color: "from-yellow-500 to-yellow-600",
  },
  {
    icon: BarChart3,
    title: "Explainable Results",
    description: "Grad-CAM heatmaps show exactly which regions influenced the AI decision.",
    color: "from-green-500 to-green-600",
  },
  {
    icon: Shield,
    title: "Clinical Grade",
    description: "Built with medical professionals in mind. Research-backed and validated.",
    color: "from-red-500 to-red-600",
  },
  {
    icon: Lock,
    title: "Secure & Private",
    description: "Your data is processed securely with enterprise-grade encryption.",
    color: "from-indigo-500 to-indigo-600",
  },
]

export default function FeaturesSection() {
  return (
    <section id="features" className="py-20 sm:py-32 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16 sm:mb-20">
          <h2 className="text-4xl sm:text-5xl font-bold text-slate-900 mb-4">Powerful Features</h2>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Everything you need for accurate and reliable histopathology analysis
          </p>
        </div>

        {/* Features grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <div
                key={index}
                className="group relative p-8 rounded-xl border border-slate-200 hover:border-slate-300 bg-white hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
              >
                {/* Gradient background on hover */}
                <div className="absolute inset-0 bg-gradient-to-br from-slate-50 to-transparent opacity-0 group-hover:opacity-100 rounded-xl transition-opacity duration-300" />

                {/* Content */}
                <div className="relative z-10">
                  <div
                    className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}
                  >
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-slate-900 mb-2">{feature.title}</h3>
                  <p className="text-slate-600 leading-relaxed">{feature.description}</p>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
