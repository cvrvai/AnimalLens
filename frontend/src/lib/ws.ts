export function createTrainingWebSocket(
  jobId: string,
  onMessage: (data: any) => void,
  onError?: (err: any) => void,
  onClose?: () => void
): () => void {
  const isBrowser = typeof window !== 'undefined';
  if (!isBrowser) return () => {};

  const host = window.location.host;
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  // If in Next.js development proxy, connect directly to FastAPI port 8088 or via proxy
  const wsUrl = process.env.NEXT_PUBLIC_WS_URL || `ws://127.0.0.1:8088/v1/train/ws/${jobId}`;

  const socket = new WebSocket(wsUrl);

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error('Error parsing WebSocket message', e);
    }
  };

  socket.onerror = (err) => {
    console.error('Training WebSocket error', err);
    if (onError) onError(err);
  };

  socket.onclose = () => {
    if (onClose) onClose();
  };

  return () => {
    if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
      socket.close();
    }
  };
}
