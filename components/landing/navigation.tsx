"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface NavigationProps {
  isScrolled: boolean
}

export default function Navigation({ isScrolled }: NavigationProps) {
  return (
    <nav
      className={cn(
        "fixed top-0 w-full z-50 transition-all duration-300",
        isScrolled ? "bg-white/80 backdrop-blur-md shadow-sm border-b border-slate-200/50" : "bg-transparent",
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 group">
          <div className="w-8 h-8 bg-gradient-to-br from-green-600 to-lime-500 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">H</span>
          </div>
          <span className="font-bold text-lg text-slate-900 group-hover:text-green-600 transition-colors">HistoAI</span>
        </Link>

        <div className="hidden md:flex items-center gap-8">
          <a href="#features" className="text-slate-600 hover:text-slate-900 transition-colors text-sm font-medium">
            Features
          </a>
          <a href="#how-it-works" className="text-slate-600 hover:text-slate-900 transition-colors text-sm font-medium">
            How It Works
          </a>
          <a href="#" className="text-slate-600 hover:text-slate-900 transition-colors text-sm font-medium">
            About
          </a>
        </div>

        <div className="flex items-center gap-3">
          <Button variant="ghost" className="text-slate-600 hover:text-slate-900">
            Sign In
          </Button>
          <Link href="/analyze">
            <Button className="bg-green-600 hover:bg-green-700 text-white">Get Started</Button>
          </Link>
        </div>
      </div>
    </nav>
  )
}
