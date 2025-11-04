"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { LucideIcon } from "lucide-react"

interface StatCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
}

export function StatCard({ title, value, icon: Icon }: StatCardProps) {
  return (
    <Card className="p-6">
        <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <Icon className="h-5 w-5 text-muted-foreground" />
        </div>
        <p className="mt-2 text-3xl font-bold">{value}</p>
    </Card>
  )
}
