"use client"

import { Card, CardContent } from "@/components/ui/card"

interface HistoryCardProps {
  analysis: {
    id: string;
    created_at: string;
    prediction: string;
    confidence: number;
    image_url: string;
  };
}

export function HistoryCard({ analysis }: HistoryCardProps) {
  const isMalignant = analysis.prediction === 'malignant';
  const formattedDate = new Date(analysis.created_at).toLocaleString();

  return (
    <Card className="overflow-hidden transition-all hover:shadow-md">
      <div className="flex items-center p-3">
        <div className="w-16 h-16 flex-shrink-0 mr-4 rounded-md overflow-hidden">
          <img src={analysis.image_url} alt={`Analysis ${analysis.id}`} className="object-cover w-full h-full" />
        </div>
        <div className="flex-grow">
          <div className="flex justify-between items-center">
            <p className={`text-md font-bold ${isMalignant ? 'text-red-500' : 'text-green-500'}`}>
              {analysis.prediction}
            </p>
            <p className="text-sm font-semibold">{(analysis.confidence * 100).toFixed(1)}%</p>
          </div>
          <p className="text-xs text-muted-foreground mt-1">{formattedDate}</p>
        </div>
      </div>
    </Card>
  );
}
