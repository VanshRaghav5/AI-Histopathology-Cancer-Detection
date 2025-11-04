"use client"

import { Upload } from "lucide-react"
import { cn } from "@/lib/utils"
import type React from "react"
import { useState, useCallback } from "react"

interface SimpleImageUploadProps {
  onImageUpload: (file: File) => void
  isLoading: boolean
}

export function SimpleImageUpload({ onImageUpload, isLoading }: SimpleImageUploadProps) {
  const [dragActive, setDragActive] = useState(false)

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setDragActive(false)
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        onImageUpload(e.dataTransfer.files[0])
      }
    },
    [onImageUpload]
  )

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onImageUpload(e.target.files[0])
    }
  }

  return (
    <div className="w-full h-full flex items-center justify-center">
      <label
        htmlFor="dropzone-file"
        className={cn(
          "relative flex flex-col items-center justify-center w-full h-full border-2 border-dashed rounded-lg cursor-pointer transition-colors",
          dragActive ? "border-primary bg-primary/5" : "border-border hover:border-primary/50",
          isLoading && "opacity-50 cursor-not-allowed"
        )}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center justify-center pt-5 pb-6">
          <Upload className="w-10 h-10 mb-4 text-primary" />
          <p className="mb-2 text-lg font-semibold text-foreground">Click to upload or drag and drop</p>
          <p className="text-sm text-muted-foreground">PNG or JPG (MAX. 10MB)</p>
        </div>
        <input id="dropzone-file" type="file" className="absolute inset-0 w-full h-full opacity-0" onChange={handleChange} disabled={isLoading} />
      </label>
    </div>
  )
}
