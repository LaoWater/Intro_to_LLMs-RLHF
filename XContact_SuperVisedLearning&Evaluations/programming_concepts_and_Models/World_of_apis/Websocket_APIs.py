import websocket
import json


# Real-time Bitcoin Price Tracking

def on_message(ws, message):
    data = json.loads(message)
    price = data['p']  # 'p' stands for price in Binance WebSocket API response
    print("BTC Price:", price)


def on_error(ws, error):
    print("Error:", error)


def on_close(ws, close_status_code, close_msg):
    print("Connection closed")


# Binance WebSocket URL for BTC/USDT ticker updates
socket_url = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# Initialize WebSocket
ws = websocket.WebSocketApp(socket_url,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)

# Run WebSocket in the main thread
ws.run_forever()
