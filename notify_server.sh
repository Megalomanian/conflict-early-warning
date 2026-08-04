#!/bin/bash
# Server-side completion notification via PushPlus (same channel as local notice)
# Usage: notify_server.sh "message"
TOKEN="7f4030a61ebf4a5f8262039154ebfea4"
CONTENT="$*"
if [ -z "$CONTENT" ]; then
    echo "用法: notify_server.sh <消息内容>"
    exit 1
fi
curl -s -X POST http://www.pushplus.plus/send \
    -H "Content-Type: application/json" \
    -d "{\"token\":\"$TOKEN\",\"title\":\"$CONTENT\",\"content\":\"$CONTENT\"}" \
    -o /dev/null -w "✅ 已发送 (HTTP %{http_code})\n"
