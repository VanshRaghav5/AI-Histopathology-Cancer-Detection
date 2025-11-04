"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Upload, X, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { cn } from "@/lib/utils"

interface ImageUploadProps {
  onImageUpload: (file: File) => Promise<void>
  isLoading?: boolean
  onError?: (error: string) => void
}

export function ImageUpload({ onImageUpload, isLoading }: ImageUploadProps) {
  const [dragActive, setDragActive] = useState(false)
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const validateFile = (file: File): string | null => {
    const maxSize = 10 * 1024 * 1024 // 10MB
    const allowedTypes = ["image/jpeg", "image/jpg", "image/png"]

    if (!allowedTypes.includes(file.type)) {
      return "Please upload only JPG or PNG files"
    }

    if (file.size > maxSize) {
      return "File size must be less than 10MB"
    }

    return null
  }

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }, [])

  const handleFile = (file: File) => {
    setError(null)

    const validationError = validateFile(file)
    if (validationError) {
      setError(validationError)
      return
    }

    if (file.type.startsWith("image/")) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)
      setSelectedFile(file)
      onImageUpload(file)
    }
  }

  const clearImage = () => {
    setSelectedImage(null)
    setSelectedFile(null)
    setError(null)
  }

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardContent className="p-4 sm:p-6">
        <div className="space-y-3 sm:space-y-4">
          <div className="text-center">
            <h2 className="font-serif text-xl sm:text-2xl font-bold text-foreground mb-2">
              Upload Histopathology Image
            </h2>
            <p className="text-muted-foreground font-sans text-sm sm:text-base">
              Upload a histopathology image for AI-powered cancer detection analysis
            </p>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="text-sm">{error}</AlertDescription>
            </Alert>
          )}

          {selectedImage ? (
            <div className="relative">
              <div className="relative rounded-lg overflow-hidden border-2 border-border">
                <img
                  src={selectedImage || "/placeholder.svg"}
                  alt="Selected histopathology image"
                  className="w-full h-48 sm:h-64 object-cover"
                />
                <Button
                  variant="destructive"
                  size="sm"
                  className="absolute top-2 right-2"
                  onClick={clearImage}
                  disabled={isLoading}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
              {selectedFile && (
                <div className="mt-2 p-3 bg-muted rounded-lg">
                  <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-2 text-sm">
                    <span className="font-medium truncate">{selectedFile.name}</span>
                    <span className="text-muted-foreground text-xs sm:text-sm">
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                    </span>
                  </div>
                </div>
              )}
              {isLoading && (
                <div className="absolute inset-0 bg-background/80 flex items-center justify-center rounded-lg">
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-5 w-5 sm:h-6 sm:w-6 border-b-2 border-primary"></div>
                    <span className="font-sans text-xs sm:text-sm text-foreground">Analyzing image...</span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div
              className={cn(
                "relative border-2 border-dashed rounded-lg p-6 sm:p-8 text-center transition-colors",
                dragActive ? "border-primary bg-primary/5" : "border-border hover:border-primary/50",
                isLoading && "opacity-50 cursor-not-allowed",
              )}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                type="file"
                accept=".jpg,.jpeg,.png"
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={isLoading}
              />
              <div className="space-y-3 sm:space-y-4">
                <div className="mx-auto w-12 h-12 sm:w-16 sm:h-16 bg-primary/10 rounded-full flex items-center justify-center">
                  <Upload className="h-6 w-6 sm:h-8 sm:w-8 text-primary" />
                </div>
                <div>
                  <p className="font-sans text-base sm:text-lg font-medium text-foreground">
                    Drop your image here, or click to browse
                  </p>
                  <p className="font-sans text-xs sm:text-sm text-muted-foreground mt-1">
                    Supports JPG, PNG files up to 10MB
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
