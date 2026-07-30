# 1. 停止舊 container
docker rm -f stock-mcp-container

# 2. 刪舊 image
docker rmi stock-mcp:latest

# 3. 重新 build
docker build --no-cache -t stock-mcp .

# 4. 啟動
docker run -d `
-p 8000:8000 `
--name stock-mcp-container `
stock-mcp
