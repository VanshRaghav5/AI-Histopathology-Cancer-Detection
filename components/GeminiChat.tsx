"use client"

import { useState } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Send } from 'lucide-react';
import { unifiedAPI } from '@/services/unified-api';

interface Message {
  text: string;
  isUser: boolean;
}

export function GeminiChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async () => {
    if (input.trim() === '') return;

    const userMessage: Message = { text: input, isUser: true };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await unifiedAPI.sendChatMessage(input);
      const botMessage: Message = { text: response, isUser: false };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage: Message = { text: 'Error: Could not get a response.', isUser: false };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[400px]">
      <div className="flex-grow overflow-y-auto p-4 space-y-4">
        {messages.map((msg, index) => (
          <div key={index} className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}>
            <div className={`px-4 py-2 rounded-lg ${msg.isUser ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-800'}`}>
              {msg.text}
            </div>
          </div>
        ))}
        {isLoading && (
            <div className='flex justify-start'>
                <div className='px-4 py-2 rounded-lg bg-gray-200 text-gray-800'>
                    Thinking...
                </div>
            </div>
        )}
      </div>
      <div className="p-4 border-t border-gray-200 flex items-center">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder="Type your message..."
          className="flex-grow"
          disabled={isLoading}
        />
        <Button onClick={handleSendMessage} disabled={isLoading} className="ml-2">
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
