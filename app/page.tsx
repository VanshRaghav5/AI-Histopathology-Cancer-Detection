"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Microscope, Upload, BrainCircuit, Eye } from "lucide-react"
import { motion } from "framer-motion"
import dynamic from 'next/dynamic'

const DarkVeil2 = dynamic(() => import('@/components/landing/dark_veil2').then((mod) => mod.default), { ssr: false })

export default function LandingPage() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.2, delayChildren: 0.3 },
    },
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { duration: 0.5 } },
  }

  return (
    <div className="flex flex-col min-h-screen bg-white text-gray-800 relative">
      {/* Fixed Background with DarkVeil */}
      <div className="fixed inset-0 z-0 bg-white">
        <div className="absolute inset-0 opacity-75">
          <DarkVeil2 />
        </div>
      </div>

      <header className="px-4 lg:px-6 h-16 flex items-center bg-transparent backdrop-blur-sm fixed top-0 w-full z-50">
        <Link href="#" className="flex items-center justify-center">
          <Microscope className="h-7 w-7 text-green-600" />
          <span className="ml-3 text-xl font-bold text-gray-900">HistoAI</span>
        </Link>
        <nav className="ml-auto flex gap-3 sm:gap-4 items-center">
          <Link href="/auth/login">
            <Button variant="ghost" className="text-sm font-medium">Login</Button>
          </Link>
          <Link href="/auth/sign-up">
            <Button className="bg-green-600 hover:bg-green-700 text-sm">Get Started</Button>
          </Link>
        </nav>
      </header>

      <main className="flex-1 relative z-10">
        {/* Hero Section */}
        <section className="relative h-screen w-full flex items-center justify-center text-center overflow-hidden">
          <motion.div
            className="container px-4 md:px-6"
            initial="hidden"
            animate="visible"
            variants={containerVariants}
          >
            <motion.h1 variants={itemVariants} className="text-4xl font-bold tracking-tighter sm:text-5xl md:text-6xl lg:text-7xl/none text-gray-900">
              AI-Powered Clarity for Cancer Diagnosis
            </motion.h1>
            <motion.p variants={itemVariants} className="mx-auto max-w-[700px] text-gray-600 md:text-xl mt-6">
              Harness the power of artificial intelligence to analyze histopathology images with unprecedented accuracy. Get instant insights with explainable AI visualizations.
            </motion.p>
            <motion.div variants={itemVariants} className="mt-8">
              <Link href="/dashboard">
                <Button size="lg" className="bg-green-600 hover:bg-green-700 text-lg px-8 py-6">
                  Get Started
                </Button>
              </Link>
            </motion.div>
          </motion.div>
        </section>

        {/* How It Works Section */}
        <section className="w-full py-16 md:py-24 lg:py-32 bg-transparent">
          <div className="container px-4 md:px-6 flex justify-center">
            <div className="w-full max-w-4xl">
              <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.5 }} transition={{ duration: 0.5 }}>
                <h2 className="text-3xl font-bold tracking-tighter text-center sm:text-4xl md:text-5xl">A Simple, Powerful Workflow</h2>
                <p className="mx-auto max-w-[700px] text-gray-600 md:text-xl mt-4 text-center">Three easy steps to transform your analysis process.</p>
              </motion.div>
              <motion.div
                className="flex flex-col sm:flex-row justify-center items-center gap-8 mt-12"
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, amount: 0.2 }}
                variants={containerVariants}
              >
                <motion.div variants={itemVariants} className="grid gap-2 text-center">
                  <div className="flex justify-center items-center mb-4">
                    <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center">
                      <Upload className="h-8 w-8 text-green-600" />
                    </div>
                  </div>
                  <h3 className="text-xl font-bold">1. Upload Image</h3>
                  <p className="text-gray-600">Securely upload your histopathology slide.</p>
                </motion.div>
                <motion.div variants={itemVariants} className="grid gap-2 text-center">
                  <div className="flex justify-center items-center mb-4">
                    <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center">
                      <BrainCircuit className="h-8 w-8 text-green-600" />
                    </div>
                  </div>
                  <h3 className="text-xl font-bold">2. AI Analysis</h3>
                  <p className="text-gray-600">Our model provides an instant prediction.</p>
                </motion.div>
                <motion.div variants={itemVariants} className="grid gap-2 text-center">
                  <div className="flex justify-center items-center mb-4">
                    <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center">
                      <Eye className="h-8 w-8 text-green-600" />
                    </div>
                  </div>
                  <h3 className="text-xl font-bold">3. Review Results</h3>
                  <p className="text-gray-600">Visualize the results with an interactive heatmap.</p>
                </motion.div>
              </motion.div>
            </div>
          </div>
        </section>

        {/* Project Origin Section */}
        <section className="w-full py-16 md:py-24 bg-transparent">
            <div className="container px-4 md:px-6 text-center">
                <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.5 }} transition={{ duration: 0.5 }}>
                    <h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">From a College Project to a Powerful Tool</h2>
                    <p className="mx-auto max-w-[700px] text-gray-600 md:text-xl mt-4">This project was born from the collaboration of three friends passionate about using AI to make a difference in medical technology.</p>
                </motion.div>
            </div>
        </section>
      </main>

      {/* Footer removed as per request */}
    </div>
  )
}
