/**
 * MHEART Frontend - Main App Component
 */

import React from 'react';
import ChatInterface from './components/ChatInterface';

function App() {
    return (
        <div className="app">
            <header className="app-header">
                <h1>MHEART</h1>
                <p>Mental Health Emotion Analysis & Response Terminal</p>
            </header>

            <main className="app-main">
                <ChatInterface />
            </main>

            <footer className="app-footer">
                <p>Privacy Notice: All processing happens locally. Your data is never sent to external servers.</p>
                <p>If you're in crisis, please call 988 (US) or your local emergency services.</p>
            </footer>
        </div>
    );
}

export default App;