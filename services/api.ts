import { GoogleGenerativeAI } from "@google/generative-ai"

const GEMINI_API_KEY = process.env.NEXT_PUBLIC_GEMINI_API_KEY || ""

if (!GEMINI_API_KEY) {
  console.error(`[v0] Missing Gemini API Key. Please set NEXT_PUBLIC_GEMINI_API_KEY in your environment variables.`)
}

// Initialize Gemini
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY)

export interface PredictionResult {
  prediction: "Cancerous" | "Non-Cancerous"
  confidence: number
  analysis: string
  processing_time?: number
}

export interface ApiError {
  message: string
  status: number
}

export const histopathologyAPI = {
  async predictCancer(imageFile: File): Promise<PredictionResult> {
    try {
      const startTime = Date.now()
      const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" })

      // Convert File to base64
      const fileBuffer = await imageFile.arrayBuffer()
      const base64String = Buffer.from(fileBuffer).toString('base64')

      // Prepare the prompt
      const prompt = "Analyze this histopathology image and determine if it shows signs of cancer. Provide your analysis in this JSON format: {prediction: 'Cancerous' or 'Non-Cancerous', confidence: number between 0 and 1, analysis: 'detailed explanation of findings'}. Base your analysis on cellular patterns, tissue organization, and any abnormal features visible in the image."

      const result = await model.generateContent([
        prompt,
        {
          inlineData: {
            mimeType: imageFile.type,
            data: base64String
          }
        }
      ])

      const response = await result.response
      const text = response.text()
      
      try {
        const analysisResult = JSON.parse(text)
        return {
          prediction: analysisResult.prediction,
          confidence: analysisResult.confidence,
          analysis: analysisResult.analysis,
          processing_time: Date.now() - startTime
        }
      } catch (parseError) {
        // If JSON parsing fails, make a best effort to extract information
        const isCancerous = text.toLowerCase().includes("cancer") || text.toLowerCase().includes("malignant")
        return {
          prediction: isCancerous ? "Cancerous" : "Non-Cancerous",
          confidence: 0.7,
          analysis: text,
          processing_time: Date.now() - startTime
        }
      }
    } catch (error) {
      console.error("[v0] Gemini API Error:", error)
      throw {
        message: error instanceof Error ? error.message : "Failed to analyze image",
        status: 500
      }
    }
  },

  async healthCheck(): Promise<boolean> {
    try {
      if (!GEMINI_API_KEY) {
        console.error("[v0] Missing Gemini API Key")
        return false
      }

      // Simple test to check if the API key is valid
      const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" })
      await model.generateContent("Test connection")
      return true
    } catch (error) {
      console.error("[v0] Gemini API health check failed:", error)
      return false
    }
  },

  getApiConfig(): { url: string; isValid: boolean } {
    return {
      url: "https://generativelanguage.googleapis.com",
      isValid: Boolean(GEMINI_API_KEY),
    }
  },

  async chatWithGemini(message: string): Promise<string> {
    try {
      const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" })
      const result = await model.generateContent(message)
      const response = await result.response
      return response.text()
    } catch (error) {
      console.error("[v0] Gemini Chat API Error:", error)
      throw new Error("Failed to get response from Gemini Chat")
    }
  },
}