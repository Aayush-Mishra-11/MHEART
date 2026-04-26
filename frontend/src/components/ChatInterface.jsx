/**
 * MHEART Frontend - Chat Interface Component
 */

import React, { useState, useRef, useEffect } from 'react';
import {
    sendChatMessage,
    uploadAudio,
    uploadVideoFrame,
    getHotlines
} from '../utils/api';

const EMOTION_COLORS = {
    happy: '#4CAF50',
    sad: '#2196F3',
    angry: '#F44336',
    fear: '#9C27B0',
    surprise: '#FF9800',
    disgust: '#795548',
    neutral: '#9E9E9E'
};

function ChatInterface() {
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [currentEmotion, setCurrentEmotion] = useState('neutral');
    const [crisisAlert, setCrisisAlert] = useState(null);
    const [hotlines, setHotlines] = useState({});

    const messagesEndRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    // Load hotlines on mount
    useEffect(() => {
        getHotlines().then(data => setHotlines(data.hotlines || {}));
    }, []);

    // Auto-scroll to bottom
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSendMessage = async () => {
        if (!inputText.trim()) return;

        const userMessage = {
            type: 'user',
            content: inputText,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInputText('');
        setIsProcessing(true);

        try {
            const response = await sendChatMessage(inputText);

            const botMessage = {
                type: 'bot',
                content: response.response,
                emotion: response.emotion_detected,
                crisisDetected: response.crisis_detected,
                crisisType: response.crisis_type,
                persona: response.persona_used,
                timestamp: new Date()
            };

            setMessages(prev => [...prev, botMessage]);
            setCurrentEmotion(response.emotion_detected);

            if (response.crisis_detected) {
                setCrisisAlert({
                    type: response.crisis_type,
                    message: 'Crisis indicators detected. Resources available.'
                });
            }
        } catch (error) {
            console.error('Error sending message:', error);
            setMessages(prev => [...prev, {
                type: 'error',
                content: 'Sorry, there was an error processing your message.',
                timestamp: new Date()
            }]);
        } finally {
            setIsProcessing(false);
        }
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorderRef.current = new MediaRecorder(stream);
            audioChunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (event) => {
                audioChunksRef.current.push(event.data);
            };

            mediaRecorderRef.current.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
                await processAudio(audioBlob);
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
        } catch (error) {
            console.error('Error starting recording:', error);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const processAudio = async (audioBlob) => {
        setIsProcessing(true);
        try {
            const response = await uploadAudio(audioBlob);

            const botMessage = {
                type: 'bot',
                content: `Audio processed - Detected emotion: ${response.emotion}`,
                emotion: response.emotion,
                crisisDetected: response.crisis_detected,
                timestamp: new Date()
            };

            setMessages(prev => [...prev, botMessage]);
            setCurrentEmotion(response.emotion);
        } catch (error) {
            console.error('Error processing audio:', error);
        } finally {
            setIsProcessing(false);
        }
    };

    const captureVideoFrame = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            const video = document.createElement('video');
            video.srcObject = stream;
            await video.play();

            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            stream.getTracks().forEach(track => track.stop());

            canvas.toBlob(async (blob) => {
                setIsProcessing(true);
                try {
                    const response = await uploadVideoFrame(blob);

                    const botMessage = {
                        type: 'bot',
                        content: `Video processed - Detected emotion: ${response.emotion}`,
                        emotion: response.emotion,
                        crisisDetected: response.crisis_detected,
                        timestamp: new Date()
                    };

                    setMessages(prev => [...prev, botMessage]);
                    setCurrentEmotion(response.emotion);
                } catch (error) {
                    console.error('Error processing video:', error);
                } finally {
                    setIsProcessing(false);
                }
            }, 'image/jpeg');
        } catch (error) {
            console.error('Error capturing video:', error);
        }
    };

    return (
        <div className="chat-container">
            {/* Emotion indicator */}
            <div className="emotion-indicator" style={{
                backgroundColor: EMOTION_COLORS[currentEmotion] || EMOTION_COLORS.neutral
            }}>
                <span>Current Emotion: {currentEmotion}</span>
            </div>

            {/* Crisis alert */}
            {crisisAlert && (
                <div className="crisis-alert">
                    <h3>Crisis Alert: {crisisAlert.type}</h3>
                    <p>{crisisAlert.message}</p>
                    <div className="hotlines">
                        <h4>Crisis Hotlines:</h4>
                        {Object.entries(hotlines).map(([country, number]) => (
                            <p key={country}>{country}: {number}</p>
                        ))}
                    </div>
                    <button onClick={() => setCrisisAlert(null)}>Dismiss</button>
                </div>
            )}

            {/* Messages */}
            <div className="messages-container">
                {messages.map((msg, index) => (
                    <div key={index} className={`message message-${msg.type}`}>
                        <div className="message-content">{msg.content}</div>
                        {msg.emotion && (
                            <div className="message-emotion" style={{
                                color: EMOTION_COLORS[msg.emotion]
                            }}>
                                Emotion: {msg.emotion}
                            </div>
                        )}
                        <div className="message-time">
                            {msg.timestamp.toLocaleTimeString()}
                        </div>
                    </div>
                ))}
                {isProcessing && (
                    <div className="message message-bot">
                        <div className="processing-indicator">Processing...</div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input area */}
            <div className="input-area">
                <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                    placeholder="Type your message..."
                    disabled={isProcessing}
                />
                <button onClick={handleSendMessage} disabled={isProcessing || !inputText.trim()}>
                    Send
                </button>

                {/* Audio recording button */}
                <button
                    className={`record-btn ${isRecording ? 'recording' : ''}`}
                    onMouseDown={startRecording}
                    onMouseUp={stopRecording}
                    disabled={isProcessing}
                >
                    {isRecording ? 'Release to send' : 'Hold to record'}
                </button>

                {/* Video capture button */}
                <button
                    onClick={captureVideoFrame}
                    disabled={isProcessing}
                    className="video-btn"
                >
                    Capture Video
                </button>
            </div>
        </div>
    );
}

export default ChatInterface;