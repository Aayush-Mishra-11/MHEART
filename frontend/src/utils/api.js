/**
 * MHEART Frontend API Utilities
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Send chat message and get response
 */
export async function sendChatMessage(message, sessionId = null) {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message,
            session_id: sessionId
        })
    });

    if (!response.ok) {
        throw new Error('Failed to send message');
    }

    return response.json();
}

/**
 * Upload audio for processing
 */
export async function uploadAudio(audioBlob, sessionId = null) {
    const formData = new FormData();
    formData.append('file', audioBlob, 'audio.wav');
    if (sessionId) {
        formData.append('session_id', sessionId);
    }

    const response = await fetch(`${API_BASE_URL}/api/audio`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error('Failed to process audio');
    }

    return response.json();
}

/**
 * Upload video frame for processing
 */
export async function uploadVideoFrame(frameBlob, sessionId = null) {
    const formData = new FormData();
    formData.append('frame', frameBlob, 'frame.jpg');
    if (sessionId) {
        formData.append('session_id', sessionId);
    }

    const response = await fetch(`${API_BASE_URL}/api/video`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        throw new Error('Failed to process video');
    }

    return response.json();
}

/**
 * Get current emotion state
 */
export async function getEmotionState() {
    const response = await fetch(`${API_BASE_URL}/api/emotion`);

    if (!response.ok) {
        throw new Error('Failed to get emotion state');
    }

    return response.json();
}

/**
 * Get crisis hotlines
 */
export async function getHotlines() {
    const response = await fetch(`${API_BASE_URL}/api/hotlines`);

    if (!response.ok) {
        throw new Error('Failed to get hotlines');
    }

    return response.json();
}

/**
 * WebSocket connection for real-time chat
 */
export function createWebSocket(sessionId, onMessage, onError) {
    const ws = new WebSocket(`${API_BASE_URL}/api/ws/chat/${sessionId}`);

    ws.onopen = () => {
        console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        onMessage(data);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (onError) onError(error);
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
    };

    return ws;
}

/**
 * Send text message via WebSocket
 */
export function sendWebSocketText(ws, content) {
    ws.send(JSON.stringify({
        type: 'text',
        content
    }));
}

/**
 * Send audio via WebSocket (base64 encoded)
 */
export function sendWebSocketAudio(ws, audioBlob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64 = reader.result.split(',')[1];
            ws.send(JSON.stringify({
                type: 'audio',
                content: base64
            }));
            resolve();
        };
        reader.onerror = reject;
        reader.readAsDataURL(audioBlob);
    });
}

/**
 * Send video via WebSocket (base64 encoded)
 */
export function sendWebSocketVideo(ws, imageBlob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64 = reader.result.split(',')[1];
            ws.send(JSON.stringify({
                type: 'video',
                content: base64
            }));
            resolve();
        };
        reader.onerror = reject;
        reader.readAsDataURL(imageBlob);
    });
}

export default {
    sendChatMessage,
    uploadAudio,
    uploadVideoFrame,
    getEmotionState,
    getHotlines,
    createWebSocket,
    sendWebSocketText,
    sendWebSocketAudio,
    sendWebSocketVideo
};