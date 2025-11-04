'use client';

import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { PredictionMode } from '@/services/unified-api';

interface Props {
  value: PredictionMode;
  onChange: (mode: PredictionMode) => void;
  backendHealthy: boolean;
  geminiHealthy: boolean;
}

export function PredictionModeSelector({ value, onChange, backendHealthy, geminiHealthy }: Props) {
  return (
    <Card className="p-4">
      <h3 className="text-sm font-semibold mb-3">Prediction Mode</h3>
      <div className="flex gap-2">
        <Button
          variant={value === 'backend' ? 'default' : 'outline'}
          onClick={() => onChange('backend')}
          disabled={!backendHealthy}
          className="flex-1"
        >
          <span className={backendHealthy ? 'text-green-500' : 'text-red-500'}>●</span>
          <span className="ml-2">Backend</span>
        </Button>
        <Button
          variant={value === 'gemini' ? 'default' : 'outline'}
          onClick={() => onChange('gemini')}
          disabled={!geminiHealthy}
          className="flex-1"
        >
          <span className={geminiHealthy ? 'text-green-500' : 'text-red-500'}>●</span>
          <span className="ml-2">Gemini</span>
        </Button>
        <Button
          variant={value === 'both' ? 'default' : 'outline'}
          onClick={() => onChange('both')}
          disabled={!backendHealthy || !geminiHealthy}
          className="flex-1"
        >
          Both
        </Button>
      </div>
    </Card>
  );
}
