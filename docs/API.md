# CryptoRL API Documentation

This document provides comprehensive API documentation for the CryptoRL trading system.

## Overview

The CryptoRL API provides RESTful endpoints for:
- **Market data** access and management
- **Trading operations** (place, modify, cancel orders)
- **Model management** (train, deploy, monitor)
- **System monitoring** (health, metrics, logs)

## Base URL

```
Development: http://localhost:8000
Production: https://api.cryptorl.com
```

## Authentication

### API Key Authentication

Include your API key in the request headers:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8000/api/v1/market-data
```

### JWT Token Authentication

For user-specific endpoints:
```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://localhost:8000/api/v1/trading/orders
```

## Rate Limits

- **Public endpoints**: 100 requests per minute
- **Authenticated endpoints**: 1000 requests per minute
- **Trading endpoints**: 60 requests per minute

## Response Format

All responses follow this structure:
```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Market Data API

### Get OHLCV Data

```http
GET /api/v1/market-data/ohlcv
```

**Parameters:**
```json
{
  "symbol": "BTCUSDT",
  "interval": "4h",
  "start": "2024-01-01T00:00:00Z",
  "end": "2024-01-15T00:00:00Z",
  "limit": 1000
}
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "open": 42000.0,
      "high": 42500.0,
      "low": 41800.0,
      "close": 42200.0,
      "volume": 1250.5
    }
  ]
}
```

### Get Real-time Ticker

```http
GET /api/v1/market-data/ticker/{symbol}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTCUSDT",
    "price": 43500.0,
    "bid": 43495.0,
    "ask": 43505.0,
    "volume_24h": 125000.5,
    "change_24h": 2.5,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Get Order Book

```http
GET /api/v1/market-data/orderbook/{symbol}
```

**Parameters:**
```json
{
  "limit": 100
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTCUSDT",
    "bids": [
      [43495.0, 1.5],
      [43490.0, 2.0],
      [43485.0, 1.8]
    ],
    "asks": [
      [43505.0, 1.2],
      [43510.0, 1.7],
      [43515.0, 2.1]
    ],
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Trading API

### Get Account Information

```http
GET /api/v1/trading/account
```

**Response:**
```json
{
  "success": true,
  "data": {
    "balance": 10000.0,
    "available_balance": 9500.0,
    "positions": [
      {
        "symbol": "BTCUSDT",
        "side": "long",
        "quantity": 0.1,
        "entry_price": 42000.0,
        "current_price": 43500.0,
        "unrealized_pnl": 150.0,
        "margin": 4200.0
      }
    ]
  }
}
```

### Place Order

```http
POST /api/v1/trading/orders
```

**Request:**
```json
{
  "symbol": "BTCUSDT",
  "side": "buy",
  "type": "limit",
  "quantity": 0.1,
  "price": 43000.0,
  "time_in_force": "gtc",
  "stop_loss": 42000.0,
  "take_profit": 44000.0
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "order_id": "12345",
    "status": "new",
    "symbol": "BTCUSDT",
    "side": "buy",
    "quantity": 0.1,
    "price": 43000.0,
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

### Cancel Order

```http
DELETE /api/v1/trading/orders/{order_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "order_id": "12345",
    "status": "cancelled",
    "cancelled_at": "2024-01-15T10:35:00Z"
  }
}
```

### Get Order Status

```http
GET /api/v1/trading/orders/{order_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "order_id": "12345",
    "status": "filled",
    "symbol": "BTCUSDT",
    "side": "buy",
    "quantity": 0.1,
    "price": 43000.0,
    "filled_quantity": 0.1,
    "filled_price": 42950.0,
    "commission": 4.3,
    "created_at": "2024-01-15T10:30:00Z",
    "filled_at": "2024-01-15T10:31:00Z"
  }
}
```

## Model Management API

### Train Model

```http
POST /api/v1/models/train
```

**Request:**
```json
{
  "algorithm": "PPO",
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "interval": "4h",
  "training_days": 90,
  "hyperparameters": {
    "learning_rate": 3e-4,
    "batch_size": 64,
    "gamma": 0.99
  },
  "use_confidence": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "job_id": "train_20240115_103000",
    "status": "running",
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

### Get Training Status

```http
GET /api/v1/models/train/{job_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "job_id": "train_20240115_103000",
    "status": "completed",
    "progress": 100,
    "model_path": "models/ppo_btc_eth_20240115",
    "metrics": {
      "sharpe_ratio": 1.8,
      "max_drawdown": 0.12,
      "win_rate": 0.65
    },
    "completed_at": "2024-01-15T12:00:00Z"
  }
}
```

### Deploy Model

```http
POST /api/v1/models/deploy
```

**Request:**
```json
{
  "model_path": "models/ppo_btc_eth_20240115",
  "environment": "production",
  "symbols": ["BTCUSDT", "ETHUSDT"]
}
```

### List Models

```http
GET /api/v1/models
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "name": "ppo_btc_eth_20240115",
      "algorithm": "PPO",
      "symbols": ["BTCUSDT", "ETHUSDT"],
      "status": "deployed",
      "sharpe_ratio": 1.8,
      "created_at": "2024-01-15T12:00:00Z"
    }
  ]
}
```

## LLM API

### Generate Confidence Vector

```http
POST /api/v1/llm/confidence
```

**Request:**
```json
{
  "symbol": "BTCUSDT",
  "date": "2024-01-15",
  "news_context": true,
  "model": "llama2:7b-chat"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTCUSDT",
    "date": "2024-01-15",
    "confidence_vector": [
      0.7,
      0.6,
      0.4,
      0.8,
      0.5,
      0.6,
      0.3
    ],
    "dimensions": [
      "fundamentals",
      "industry_condition",
      "geopolitics",
      "macroeconomics",
      "technical_analysis",
      "market_sentiment",
      "risk_assessment"
    ],
    "generated_at": "2024-01-15T10:30:00Z"
  }
}
```

### Batch Confidence Generation

```http
POST /api/v1/llm/confidence/batch
```

**Request:**
```json
{
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "start_date": "2024-01-01",
  "end_date": "2024-01-15",
  "model": "llama2:7b-chat"
}
```

## Monitoring API

### Health Check

```http
GET /api/v1/health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "services": {
      "database": "connected",
      "redis": "connected",
      "binance": "connected"
    }
  }
}
```

### Get Metrics

```http
GET /api/v1/metrics
```

**Response:**
```json
{
  "success": true,
  "data": {
    "trading": {
      "total_pnl": 1250.50,
      "daily_pnl": 45.20,
      "win_rate": 0.65,
      "sharpe_ratio": 1.8,
      "max_drawdown": 0.12
    },
    "system": {
      "active_positions": 2,
      "api_calls": 1500,
      "model_accuracy": 0.68,
      "uptime": "99.9%"
    }
  }
}
```

### Get Logs

```http
GET /api/v1/logs
```

**Parameters:**
```json
{
  "level": "ERROR",
  "start_time": "2024-01-15T09:00:00Z",
  "end_time": "2024-01-15T10:30:00Z",
  "limit": 100
}
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2024-01-15T10:25:30Z",
      "level": "ERROR",
      "message": "Order placement failed",
      "context": {
        "symbol": "BTCUSDT",
        "error": "Insufficient balance"
      }
    }
  ]
}
```

## System API

### Get System Status

```http
GET /api/v1/system/status
```

**Response:**
```json
{
  "success": true,
  "data": {
    "version": "1.2.3",
    "uptime": "15d 3h 45m",
    "cpu_usage": 45.2,
    "memory_usage": 65.8,
    "disk_usage": 78.5,
    "active_connections": 150,
    "database_status": "healthy",
    "model_status": "deployed"
  }
}
```

### Restart Services

```http
POST /api/v1/system/restart
```

**Request:**
```json
{
  "services": ["trading", "training"],
  "force": false
}
```

### Update Configuration

```http
PUT /api/v1/system/config
```

**Request:**
```json
{
  "trading": {
    "max_positions": 5,
    "max_position_size": 0.2,
    "stop_loss": 0.05
  },
  "risk": {
    "max_drawdown": 0.15,
    "var_threshold": 0.05
  }
}
```

## Error Responses

### Standard Error Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "symbol": "Symbol is required"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Codes
- `VALIDATION_ERROR`: Invalid input parameters
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INSUFFICIENT_BALANCE`: Not enough balance for trading
- `MARKET_CLOSED`: Market is currently closed
- `MODEL_NOT_FOUND`: Requested model doesn't exist
- `TRAINING_IN_PROGRESS`: Model training is still running

## WebSocket API

### Real-time Market Data

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/market-data');

// Subscribe to symbols
ws.send(JSON.stringify({
  action: 'subscribe',
  symbols: ['BTCUSDT', 'ETHUSDT']
}));

// Receive updates
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Price update:', data);
};
```

### Real-time Trading Updates

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/trading');

ws.onmessage = function(event) {
  const update = JSON.parse(event.data);
  
  switch(update.type) {
    case 'order_filled':
      console.log('Order filled:', update.data);
      break;
    case 'position_update':
      console.log('Position updated:', update.data);
      break;
    case 'pnl_update':
      console.log('PnL updated:', update.data);
      break;
  }
};
```

## SDK Examples

### Python SDK

```python
import requests

class CryptoRLClient:
    def __init__(self, api_key, base_url="http://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def get_market_data(self, symbol, interval, start, end):
        params = {
            "symbol": symbol,
            "interval": interval,
            "start": start,
            "end": end
        }
        response = requests.get(
            f"{self.base_url}/api/v1/market-data/ohlcv",
            params=params,
            headers=self.headers
        )
        return response.json()
    
    def place_order(self, symbol, side, quantity, price=None):
        data = {
            "symbol": symbol,
            "side": side,
            "type": "market" if price is None else "limit",
            "quantity": quantity
        }
        if price:
            data["price"] = price
        
        response = requests.post(
            f"{self.base_url}/api/v1/trading/orders",
            json=data,
            headers=self.headers
        )
        return response.json()

# Usage
client = CryptoRLClient("your_api_key")
data = client.get_market_data("BTCUSDT", "4h", "2024-01-01", "2024-01-15")
order = client.place_order("BTCUSDT", "buy", 0.1)
```

### JavaScript SDK

```javascript
class CryptoRLClient {
  constructor(apiKey, baseUrl = 'http://localhost:8000') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
    this.headers = {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    };
  }

  async getMarketData(symbol, interval, start, end) {
    const params = new URLSearchParams({ symbol, interval, start, end });
    const response = await fetch(`${this.baseUrl}/api/v1/market-data/ohlcv?${params}`, {
      headers: this.headers
    });
    return response.json();
  }

  async placeOrder(symbol, side, quantity, price = null) {
    const data = { symbol, side, quantity };
    if (price) data.price = price;
    
    const response = await fetch(`${this.baseUrl}/api/v1/trading/orders`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(data)
    });
    return response.json();
  }

  async subscribeToMarketData(symbols, callback) {
    const ws = new WebSocket(`${this.baseUrl.replace('http', 'ws')}/ws/market-data`);
    
    ws.onopen = () => {
      ws.send(JSON.stringify({ action: 'subscribe', symbols }));
    };
    
    ws.onmessage = (event) => {
      callback(JSON.parse(event.data));
    };
    
    return ws;
  }
}

// Usage
const client = new CryptoRLClient('your_api_key');
const data = await client.getMarketData('BTCUSDT', '4h', '2024-01-01', '2024-01-15');
const order = await client.placeOrder('BTCUSDT', 'buy', 0.1);
```