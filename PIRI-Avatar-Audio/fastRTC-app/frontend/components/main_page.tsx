"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { AIVoiceInput } from "@/components/ui/ai_voice_chat";
import { WebRTCClient } from "@/lib/webrtc-client";
import { motion, AnimatePresence } from "framer-motion";

// Simple chat bubble for user/assistant
function ChatBubble({ type, text }: { type: string; text: string }) {
    const isUser = type === "stt";
    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}
        >
            <div
                className={`rounded-2xl px-4 py-3 max-w-[80%] shadow-sm ${
                    isUser
                        ? "bg-blue-500 text-white rounded-br-none"
                        : "bg-white text-gray-900 rounded-bl-none border border-gray-100"
                }`}
            >
                <p className="text-sm leading-relaxed">{text}</p>
            </div>
        </motion.div>
    );
}

export function BackgroundCircleProvider() {
    const [isConnected, setIsConnected] = useState(false);
    const [webrtcClient, setWebrtcClient] = useState<WebRTCClient | null>(null);
    const [audioLevel, setAudioLevel] = useState(0);
    const [chatMessages, setChatMessages] = useState<{ type: string; text: string }[]>([]);
    const audioRef = useRef<HTMLAudioElement>(null);
    const clientRef = useRef<WebRTCClient | null>(null);
    const outputDeviceIdRef = useRef<string | undefined>(undefined);
    const chatBottomRef = useRef<HTMLDivElement>(null);
    const [currentVisemes, setCurrentVisemes] = useState<any[]>([]);
    const [isAnimating, setIsAnimating] = useState(false);

    const [webrtcId] = useState(() => Math.random().toString(36).substring(7));
    // Scroll to bottom on new chat message
    useEffect(() => {
        chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [chatMessages]);

    // Memoize callbacks to prevent recreation on each render
    const handleConnected = useCallback(() => setIsConnected(true), []);
    const handleDisconnected = useCallback(() => setIsConnected(false), []);

    const handleAudioStream = useCallback((stream: MediaStream) => {
        if (!audioRef.current) return;

        audioRef.current.srcObject = stream;

        if ('setSinkId' in HTMLAudioElement.prototype && outputDeviceIdRef.current) {
            try {
                (audioRef.current as any).setSinkId(outputDeviceIdRef.current)
                    .catch((err: any) => {
                        console.error('Error setting audio output device:', err);
                    });
            } catch (err) {
                console.error('Error applying setSinkId:', err);
            }
        }
    }, []);

    const handleAudioLevel = useCallback((level: number) => {
        setAudioLevel(prev => prev * 0.7 + level * 0.3);
    }, []);

    // NEW: Handle incoming chat messages (STT & LLM)
    const handleMessage = useCallback((message: any) => {
        if (message?.type && message?.text) {
            setChatMessages(prev => [...prev, { type: message.type, text: message.text }]);
        }
    }, []);

    const handleVisemeData = useCallback((visemeData: any) => {
        setCurrentVisemes(visemeData.visemes);
        setIsAnimating(true);

        // Optional: Auto-stop animation after viseme duration
        const totalDuration = visemeData.chunk_duration * 1000; // Convert to ms
        setTimeout(() => {
            setIsAnimating(false);
            setCurrentVisemes([]);
        }, totalDuration);

        // You can also trigger your avatar animation here
        // updateAvatarMouth(visemeData.visemes);
    }, []);

    useEffect(() => {
        // Initialize WebRTC client with memoized callbacks
        const client = new WebRTCClient({
            onConnected: handleConnected,
            onDisconnected: handleDisconnected,
            onAudioStream: handleAudioStream,
            onAudioLevel: handleAudioLevel,
            onMessage: handleMessage, // <-- Add this!
            webrtcId: webrtcId,
        });

        setWebrtcClient(client);
        clientRef.current = client;

        return () => {
            client.disconnect();
            clientRef.current = null;
        };
    }, [handleConnected, handleDisconnected, handleAudioStream, handleAudioLevel, handleMessage]);

    // useEffect(() => {
    //     if (!webrtcId) return;
    //     const eventSource = new EventSource(`http://localhost:8081/updates?webrtc_id=${webrtcId}`);
    //     eventSource.onmessage = (event) => {
    //         try {
    //             const data = JSON.parse(event.data);
    //             if (data?.type === "stt" || data?.type === "llm") {
    //                 setChatMessages((prev) => [...prev, data]);
    //             }
    //         } catch (err) {
    //             console.error("SSE parse error", err);
    //         }
    //     };
    //     eventSource.onerror = (err) => {
    //         console.error("SSE error", err);
    //     };
    //     return () => eventSource.close();
    // }, [webrtcId]);

    useEffect(() => {
    if (!webrtcId) return;
    const eventSource = new EventSource(`http://localhost:8081/updates?webrtc_id=${webrtcId}`);
    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data?.type === "stt" || data?.type === "llm") {
                setChatMessages((prev) => [...prev, data]);
            } else if (data?.type === "visemes") {
                // Handle viseme data for avatar animation
                console.log("Received visemes:", data);
                handleVisemeData(data);
            }
        } catch (err) {
            console.error("SSE parse error", err);
        }
    };
    eventSource.onerror = (err) => {
        console.error("SSE error", err);
    };
    return () => eventSource.close();
}, [webrtcId]);

    const handleStart = useCallback(() => {
        if (clientRef.current) {
            clientRef.current.connect().catch(error => {
                console.error('Failed to connect:', error);
            });
        }
    }, []);

    const handleStop = useCallback(() => {
        if (clientRef.current) {
            clientRef.current.disconnect();
        }
    }, []);

    // Handle device change
    const handleDeviceChange = useCallback((deviceId: string, type: 'input' | 'output') => {
        if (!clientRef.current) return;

        if (type === 'input') {
            clientRef.current.setAudioInputDevice(deviceId);
        } else if (type === 'output') {
            clientRef.current.setAudioOutputDevice(deviceId);
            outputDeviceIdRef.current = deviceId;

            if (audioRef.current && audioRef.current.srcObject && 'setSinkId' in HTMLAudioElement.prototype) {
                try {
                    (audioRef.current as any).setSinkId(deviceId)
                        .catch((err: any) => {
                            console.error('Error setting audio output device:', err);
                        });
                } catch (err) {
                    console.error('Error applying setSinkId:', err);
                }
            }
        }
    }, []);

    return (
        <div className="relative w-full h-screen bg-gradient-to-br from-gray-50 to-blue-50 overflow-hidden">
            {/* Decorative background elements */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute -top-1/4 -right-1/4 w-1/2 h-1/2 bg-blue-100 rounded-full mix-blend-multiply filter blur-3xl opacity-70"></div>
                <div className="absolute -bottom-1/4 -left-1/4 w-1/2 h-1/2 bg-indigo-100 rounded-full mix-blend-multiply filter blur-3xl opacity-70"></div>
            </div>

            {/* Main content */}
            <div className="relative z-10 h-full flex flex-col max-w-4xl mx-auto">
                {/* Header */}
                <header className="px-6 py-4">
                    <h1 className="text-2xl font-bold text-gray-800">Voice Assistant</h1>
                    <div className={`inline-flex items-center mt-1 text-sm ${
                        isConnected ? 'text-green-500' : 'text-gray-400'
                    }`}>
                        <span className={`w-2 h-2 rounded-full mr-2 ${
                            isConnected ? 'bg-green-500' : 'bg-gray-400'
                        }`}></span>
                        {isConnected ? 'Connected' : 'Disconnected'}
                    </div>
                </header>

                {/* Chat container */}
                <div className="flex-1 overflow-hidden px-4">
                    <div className="h-full flex flex-col">
                        {/* Chat messages */}
                        <div className="flex-1 overflow-y-auto py-4 px-2 space-y-3">
                            {chatMessages.length === 0 ? (
                                <motion.div 
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="h-full flex flex-col items-center justify-center text-center text-gray-400"
                                >
                                    <div className="w-16 h-16 mb-4 rounded-full bg-white shadow-sm flex items-center justify-center">
                                        <svg className="w-8 h-8 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                                        </svg>
                                    </div>
                                    <p className="text-lg font-medium text-gray-500">Start a conversation</p>
                                    <p className="text-sm mt-1">Press and hold the button below to speak</p>
                                </motion.div>
                            ) : (
                                chatMessages.map((msg, idx) => (
                                    <ChatBubble key={idx} type={msg.type} text={msg.text} />
                                ))
                            )}
                            <div ref={chatBottomRef} />
                        </div>

                        {/* Audio level meter */}
                        <motion.div 
                            className="px-2 mb-4"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 }}
                        >
                            <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden">
                                <motion.div
                                    className="h-full bg-blue-500 rounded-full"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${Math.min(audioLevel * 100, 100)}%` }}
                                    transition={{ type: "spring", damping: 15 }}
                                />
                            </div>
                        </motion.div>

                        {/*/!* Viseme debug display *!/*/}
                        {/*{currentVisemes.length > 0 && (*/}
                        {/*    <motion.div*/}
                        {/*        className="px-2 mb-2"*/}
                        {/*        initial={{ opacity: 0 }}*/}
                        {/*        animate={{ opacity: 1 }}*/}
                        {/*    >*/}
                        {/*        <div className="text-xs text-gray-500 mb-1">Current Visemes:</div>*/}
                        {/*        <div className="flex flex-wrap gap-1">*/}
                        {/*            {currentVisemes.map((viseme, idx) => (*/}
                        {/*                <span*/}
                        {/*                    key={idx}*/}
                        {/*                    className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs"*/}
                        {/*                >*/}
                        {/*                    {viseme.viseme}*/}
                        {/*                </span>*/}
                        {/*            ))}*/}
                        {/*        </div>*/}
                        {/*    </motion.div>*/}
                        {/*)}*/}

                    </div>
                </div>

                {/* Voice input controls */}
                <div className="p-6 pt-2">
                    <div className="flex justify-center">
                        <AIVoiceInput
                            onStart={handleStart}
                            onStop={handleStop}
                            isConnected={isConnected}
                        />
                    </div>
                </div>
            </div>

            {/* Hidden audio element for playback */}
            <audio ref={audioRef} autoPlay hidden />
        </div>
    );
}

export default { BackgroundCircleProvider };