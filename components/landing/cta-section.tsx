"use client"

import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function CTASection() {
  return (
    <section className="py-20 sm:py-32 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="relative rounded-2xl overflow-hidden">
          {/* Background gradient */}
          <div className="absolute inset-0 bg-gradient-to-r from-green-600 to-lime-500" />
          <div className="absolute inset-0 bg-grid-pattern opacity-10" />

          {/* Content */}
          <div className="relative z-10 px-8 sm:px-12 py-16 sm:py-20 text-center">
            <h2 className="text-4xl sm:text-5xl font-bold text-white mb-4">Ready to Transform Your Analysis?</h2>
            <p className="text-lg text-green-50 mb-8 max-w-2xl mx-auto">
              Join researchers and medical professionals using HistoAI for accurate, explainable cancer detection.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link href="/analyze">
                <Button className="bg-white hover:bg-slate-100 text-green-600 px-8 py-6 text-lg rounded-full group">
                  Start Free Analysis
                  <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
              <Button
                variant="outline"
                className="px-8 py-6 text-lg rounded-full border-white text-white hover:bg-white/10 bg-transparent"
              >
                Learn More
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
