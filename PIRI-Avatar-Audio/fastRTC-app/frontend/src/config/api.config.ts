/**
 * API Configuration
 * Centralized configuration for all API endpoints
 */

export const API_CONFIG = {
  // Base URL for the backend server
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001',
  
  // WebSocket URLs
  WS_BASE_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8001',
  
  // API Endpoints
  ENDPOINTS: {
    // Avatar endpoints
    AVATAR: {
      WEBSOCKET: '/api/avatar/ws/avatar',
      TRIGGER_VISEME: '/api/avatar/trigger_viseme',
      STATUS: '/api/avatar/avatar_status',
      RESET: '/api/avatar/reset_avatar',
    },
    
    // WebRTC endpoints
    WEBRTC: {
      OFFER: '/webrtc/offer',
      UPDATES: '/updates',
    },
    
    // Health check
    HEALTH: '/health',
  },
  
  // Timeout configurations
  TIMEOUTS: {
    DEFAULT: 30000, // 30 seconds
    WEBSOCKET_RECONNECT: 5000, // 5 seconds
  },
};

// Helper functions
export const getApiUrl = (endpoint: string): string => {
  return `${API_CONFIG.BASE_URL}${endpoint}`;
};

export const getWsUrl = (endpoint: string): string => {
  return `${API_CONFIG.WS_BASE_URL}${endpoint}`;
};
