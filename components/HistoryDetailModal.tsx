"use client"

import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogClose } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"

interface HistoryDetailModalProps {
  analysis: {
    id: string;
    created_at: string;
    prediction: string;
    confidence: number;
    image_url: string;
    heatmap?: string;
  };
  children: React.ReactNode;
}

export function HistoryDetailModal({ analysis, children }: HistoryDetailModalProps) {
  return (
    <Dialog>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle>Analysis Details</DialogTitle>
        </DialogHeader>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
          <div>
            <h3 className="text-lg font-semibold mb-2">Original Image</h3>
            <img src={analysis.image_url} alt="Original" className="rounded-lg border" />
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-2">Grad-CAM Heatmap</h3>
            {analysis.heatmap ? (
              <img src={analysis.heatmap} alt="Heatmap" className="rounded-lg border" />
            ) : (
              <p className="text-muted-foreground">No heatmap available for this analysis.</p>
            )}
          </div>
        </div>
        <div className="mt-4">
            <p><strong>Prediction:</strong> {analysis.prediction}</p>
            <p><strong>Confidence:</strong> {(analysis.confidence * 100).toFixed(2)}%</p>
            <p><strong>Date:</strong> {new Date(analysis.created_at).toLocaleString()}</p>
        </div>
        <DialogClose asChild>
          <Button type="button" variant="secondary" className="mt-4">Close</Button>
        </DialogClose>
      </DialogContent>
    </Dialog>
  )
}
