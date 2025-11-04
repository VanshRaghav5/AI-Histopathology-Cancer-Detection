import axios from 'axios';
import { GoogleGenerativeAI } from "@google/generative-ai";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
const GEMINI_API_KEY = process.env.NEXT_PUBLIC_GEMINI_API_KEY || '';
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

export type PredictionMode = 'backend' | 'gemini' | 'both';

export interface UnifiedPredictionResult {
  prediction: string;
  confidence: number;
  analysis?: string;
  probabilities?: { benign: number; malignant: number };
  source: 'backend' | 'gemini' | 'combined';
  processing_time: number;
  heatmap?: string;
}

export const unifiedAPI = {
  async predictCancer(file: File, mode: PredictionMode = 'backend'): Promise<UnifiedPredictionResult> {
    const startTime = Date.now();
    
    if (mode === 'backend') {
      return await this.predictWithBackend(file, startTime);
    } else if (mode === 'gemini') {
      return await this.predictWithGemini(file, startTime);
    } else {
      const [backendResult, geminiResult] = await Promise.all([
        this.predictWithBackend(file, startTime).catch(() => null),
        this.predictWithGemini(file, startTime).catch(() => null)
      ]);
      return this.combineResults(backendResult, geminiResult, startTime);
    }
  },

  async predictWithBackend(file: File, startTime: number): Promise<UnifiedPredictionResult> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${BACKEND_URL}/predict-with-gradcam`, formData);
    return {
      ...response.data,
      source: 'backend',
      processing_time: Date.now() - startTime
    };
  },

  async predictWithGemini(file: File, startTime: number): Promise<UnifiedPredictionResult> {
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    const fileBuffer = await file.arrayBuffer();
    const base64String = Buffer.from(fileBuffer).toString('base64');
    
    const prompt = `Analyze this histopathology image. Return JSON: {"prediction": "Cancerous" or "Non-Cancerous", "confidence": 0-1, "analysis": "explanation"}`;
    
    const result = await model.generateContent([
      prompt,
      { inlineData: { mimeType: file.type, data: base64String } }
    ]);
    
    const analysisResult = JSON.parse(result.response.text());
    return {
      prediction: analysisResult.prediction === 'Cancerous' ? 'malignant' : 'benign',
      confidence: analysisResult.confidence,
      analysis: analysisResult.analysis,
      source: 'gemini',
      processing_time: Date.now() - startTime
    };
  },

  combineResults(backend: any, gemini: any, startTime: number): UnifiedPredictionResult {
    if (!backend) return { ...gemini, source: 'gemini' };
    if (!gemini) return { ...backend, source: 'backend' };
    return {
      prediction: backend.prediction,
      confidence: (backend.confidence + gemini.confidence) / 2,
      analysis: gemini.analysis,
      probabilities: backend.probabilities,
      source: 'combined',
      processing_time: Date.now() - startTime
    };
  },

  async checkBackendHealth(): Promise<boolean> {
    try {
      const response = await axios.get(`${BACKEND_URL}/health`, { timeout: 5000 });
      return response.data.status === 'healthy';
    } catch {
      return false;
    }
  },

  async checkGeminiHealth(): Promise<boolean> {
    try {
      if (!GEMINI_API_KEY) return false;
      const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
      await model.generateContent("Test");
      return true;
    } catch {
      return false;
    }
  },

  async sendChatMessage(message: string): Promise<string> {
    if (!GEMINI_API_KEY) {
      throw new Error("Gemini API key not configured.");
    }

    const systemPrompt = `
      You are an intelligent assistant for HistoAI, a web application for histopathology image analysis.
      Your purpose is to answer user questions about the project, its technology, and its functionality.

      Key details about HistoAI:
      - **Purpose**: To provide AI-powered cancer detection from histopathology images.
      - **Core Feature**: Users can upload an image, and the system will predict whether it is benign or malignant.
      - **Visualization**: It uses Grad-CAM to generate a heatmap, showing which parts of the image the model focused on for its prediction.
      - **Technology Stack**:
        - **Frontend**: Next.js and Tailwind CSS.
        - **Backend**: FastAPI (Python).
        - **AI Model**: A PyTorch-based Convolutional Neural Network (CNN), specifically a fine-tuned MobileNetV2.
        - **Database**: Supabase for user authentication and storing analysis history.

      When answering, be concise and helpful. Keep your answers short and to the point. If you don't know the answer, say so.
    `;

    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash", systemInstruction: systemPrompt });
    const result = await model.generateContent(message);
    return result.response.text();
  }
};
