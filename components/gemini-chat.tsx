"use client"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send, Bot, User } from "lucide-react"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Separator } from "@/components/ui/separator"

interface Message {
  role: "user" | "model"
  content: string
}

interface GeminiChatProps {
  onSendMessage: (message: string) => Promise<string>
  isLoading: boolean
}

export function GeminiChat({ onSendMessage, isLoading }: GeminiChatProps) {
  const [input, setInput] = useState("")
  const [messages, setMessages] = useState<Message[]>([])
  const scrollAreaRef = useRef<HTMLDivElement>(null)

  const handleSend = async () => {
    if (input.trim() === "") return

    const userMessage: Message = { role: "user", content: input }
    setMessages((prevMessages) => [...prevMessages, userMessage])
    setInput("")

    try {
      const modelResponse = await onSendMessage(input)
      setMessages((prevMessages) => [...prevMessages, { role: "model", content: modelResponse }])
    } catch (error) {
      console.error("Error sending message to Gemini:", error)
      setMessages((prevMessages) => [
        ...prevMessages,
        { role: "model", content: "Sorry, I couldn't process that. Please try again." },
      ])
    }
  }

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTo({
        top: scrollAreaRef.current.scrollHeight,
        behavior: "smooth",
      })
    }
  }, [messages])

  return (
    <Card className="flex flex-col h-[500px]">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bot className="h-6 w-6 text-primary" />
          Gemini Chat
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 overflow-hidden p-0">
        <ScrollArea className="h-full p-6" ref={scrollAreaRef}>
          <div className="space-y-4">
            {messages.length === 0 && (
              <div className="text-center text-muted-foreground">
                Start a conversation with Gemini AI!
              </div>
            )}
            {messages.map((message, index) => (
              <div key={index} className={`flex items-start gap-3 ${message.role === "user" ? "justify-end" : ""}`}>
                {message.role === "model" && (
                  <Avatar className="h-8 w-8">
                    <AvatarFallback>
                      <Bot className="h-5 w-5" />
                    </AvatarFallback>
                  </Avatar>
                )}
                <div
                  className={`max-w-[70%] p-3 rounded-lg ${
                    message.role === "user"
                      ? "bg-primary text-primary-foreground rounded-br-none"
                      : "bg-muted rounded-bl-none"
                  }`}
                >
                  <p className="text-sm">{message.content}</p>
                </div>
                {message.role === "user" && (
                  <Avatar className="h-8 w-8">
                    <AvatarFallback>
                      <User className="h-5 w-5" />
                    </AvatarFallback>
                  </Avatar>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="flex items-start gap-3">
                <Avatar className="h-8 w-8">
                  <AvatarFallback>
                    <Bot className="h-5 w-5" />
                  </AvatarFallback>
                </Avatar>
                <div className="max-w-[70%] p-3 rounded-lg bg-muted rounded-bl-none animate-pulse">
                  <div className="h-4 w-24 bg-gray-300 rounded"></div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
      <CardFooter className="p-4 border-t">
        <div className="flex w-full space-x-2">
          <Input
            placeholder="Type your message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === "Enter" && !isLoading) {
                handleSend()
              }
            }}
            disabled={isLoading}
          />
          <Button onClick={handleSend} disabled={isLoading}>
            <Send className="h-4 w-4" />
            <span className="sr-only">Send</span>
          </Button>
        </div>
      </CardFooter>
    </Card>
  )
}