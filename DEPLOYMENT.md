# memory-milvus 插件部署和配置计划

## 阶段 1：插件构建

### 1.1 本地构建
```bash
# 在 /tmp/clawmemory-milvus 目录
cd /tmp/clawmemory-milvus

# 安装依赖
npm install

# 构建 TypeScript
npm run build

# 验证构建产物
ls -la dist/
# 应该看到：
# - index.js
# - index.d.ts
# - config.js
# - config.d.ts
# - ...
```

---

## 阶段 2：安装插件到 OpenClaw

### 2.1 使用 OpenClaw 插件管理器安装
```bash
# 方案 A：从本地路径安装
openclaw plugins install /tmp/clawmemory-milvus

# 方案 B：从 GitHub 安装（如果已推送）
openclaw plugins install chunliu/clawmemory-milvus

# 方案 C：从 npm 安装（如果已发布）
openclaw plugins install memory-milvus
```

### 2.2 验证插件安装
```bash
# 列出已安装的插件
openclaw plugins list

# 期望看到：
# memory-milvus (installed)
```

---

## 阶段 3：配置 Milvus 连接

### 3.1 获取 Milvus 服务信息
```bash
# 假设 Milvus 服务信息如下（请替换为实际信息）：
MILVUS_ADDRESS="your-milvus-host:19530"
MILVUS_USERNAME=""  # 如果启用了认证
MILVUS_PASSWORD=""  # 如果启用了认证
```

### 3.2 验证 Milvus 连接
```bash
# 使用 Python SDK 验证（可选）
pip install pymilvus
python3 << 'EOF'
from pymilvus import connections, utility

# 替换为实际的 Milvus 地址
connections.connect("default", host="your-milvus-host", port="19530")
print("Connected to Milvus successfully!")

# 获取 Milvus 版本
print(f"Milvus version: {utility.get_server_version()}")
EOF
```

---

## 阶段 4：配置 OpenClaw

### 4.1 定位 OpenClaw 配置文件
```bash
# 查找 OpenClaw 配置
find /home/openclaw -name "config.json" -o -name "openclaw.json" 2>/dev/null

# 通常位置：
# - ~/.openclaw/config.json
# - ~/.openclaw/workspace-fslark/config.json
# - /home/openclaw/.openclaw/config.json
```

### 4.2 备份现有配置
```bash
# 备份配置文件
cp ~/.openclaw/config.json ~/.openclaw/config.json.backup
```

### 4.3 添加插件配置
```json
{
  "plugins": {
    "slots": {
      "memory": "memory-milvus"
    },
    "entries": {
      "memory-milvus": {
        "milvus": {
          "address": "your-milvus-host:19530",
          "prefix": "openclaw-memory",
          "username": "",
          "password": ""
        },
        "embedding": {
          "provider": "openai",
          "model": "text-embedding-3-small",
          "apiKey": "your-openai-api-key",
          "dimensions": 1536
        },
        "search": {
          "maxResults": 5,
          "minScore": 0.5,
          "hybrid": {
            "enabled": true,
            "vectorWeight": 0.7,
            "textWeight": 0.3
          },
          "mmr": {
            "enabled": false,
            "lambda": 0.7
          },
          "temporalDecay": {
            "enabled": false,
            "halfLifeDays": 30
          }
        },
        "sync": {
          "onSearch": true,
          "onSessionStart": true,
          "watch": true,
          "debounceMs": 1500
        },
        "cache": {
          "enabled": true,
          "maxEntries": 50000
        },
        "sources": ["memory"]
      }
    }
  }
}
```

### 4.4 配置说明
```json
{
  "plugins": {
    "slots": {
      // 关键：指定使用 memory-milvus 插件
      "memory": "memory-milvus"
    },
    "entries": {
      "memory-milvus": {
        // Milvus 连接配置
        "milvus": {
          "address": "your-milvus-host:19530",  // Milvus 地址
          "prefix": "openclaw-memory",             // Collection 前缀
          "username": "",                          // 认证（可选）
          "password": ""                           // 认证（可选）
        },
        // 嵌入提供商配置
        "embedding": {
          "provider": "openai",                   // 嵌入提供商
          "model": "text-embedding-3-small",      // 模型
          "apiKey": "your-openai-api-key",         // API Key
          "dimensions": 1536                       // 向量维度
        },
        // 搜索配置
        "search": {
          "maxResults": 5,                         // 最大结果数
          "minScore": 0.5,                         // 最小分数
          "hybrid": {
            "enabled": true,                       // 启用混合搜索
            "vectorWeight": 0.7,                   // 向量权重
            "textWeight": 0.3                      // BM25 权重
          },
          "mmr": {
            "enabled": false,                      // 启用 MMR
            "lambda": 0.7                          // MMR lambda
          },
          "temporalDecay": {
            "enabled": false,                      // 启用时间衰减
            "halfLifeDays": 30                     // 半衰期（天）
          }
        },
        // 同步配置
        "sync": {
          "onSearch": true,                        // 搜索时同步
          "onSessionStart": true,                   // 会话开始时同步
          "watch": true,                            // 监控文件
          "debounceMs": 1500                        // 防抖延迟（ms）
        },
        // 缓存配置
        "cache": {
          "enabled": true,                          // 启用缓存
          "maxEntries": 50000                       // 最大缓存条目
        },
        // 数据源配置
        "sources": ["memory"]                      // 索引的源
      }
    }
  }
}
```

---

## 阶段 5：验证插件加载

### 5.1. 重启 OpenClaw
```bash
# 方案 A：如果 OpenClaw 作为服务运行
systemctl restart openclaw

# 方案 B：如果 OpenClaw 作为进程运行
pkill -f openclaw
openclaw start

# 方案 C：如果使用 PM2
pm2 restart openclaw
```

### 5.2 检查插件加载日志
```bash
# 查看 OpenClaw 日志
tail -f ~/.openclaw/logs/openclaw.log | grep -i "memory-milvus"

# 期望看到：
# [INFO] memory-milvus: plugin registered
# [INFO] memory-milvus: initializing (address: your-milvus-host:19530, prefix: openclaw-memory)
# [INFO] memory-milvus: file watcher started
# [INFO] memory-milvus: ready
```

### 5.3 验证插件状态
```bash
# 使用 OpenClaw CLI 检查插件
openclaw plugins list

# 期望看到：
# memory-milvus (enabled)
```

### 5.4 测试插件 CLI 命令
```bash
# 测试 memory status 命令
openclaw memory status

# 期望输出：
# Agent ID: fslark
# Collection: openclaw-memory-fslark
# {
#   "backend": "milvus",
#   "provider": "milvus",
#   "files": 0,
#   "chunks": 0,
#   "dirty": false,
#   "collectionName": "openclaw-memory-fslark",
#   ...
# }
```

---

## 阶段 6：验证 Milvus Collection

### 6.1 检查 Collection 是否创建
```python
# 使用 Python SDK 检查
from pymilvus import connections, utility

# 替换为实际的 Milvus 地址
connections.connect("default", host="your-milvus-host", port="19530")

# 列出所有 collections
collections = utility.list_collections()
print(f"Collections: {collections}")

# 期望看到：
# ['openclaw-memory-fslark']
```

### 6.2 检查 Collection Schema
```python
from pymilvus import Collection

collection = Collection("openclaw-memory-fslark")
collection.load()

# 获取 schema
schema = collection.schema
print(f"Schema: {schema}")

# 期望看到：
# - id (VarChar, primary key)
# - path (VarChar)
# - start_line (Int64)
# - end_line (Int64)
# - text (VarChar, with analyzer)
# - embedding (FloatVector)
# - model (VarChar)
# - source (VarChar)
```

---

## 阶段 7：测试工具可用性

### 7.1 测试 memory_search 工具
```bash
# 在 OpenClaw 交互会话中测试
# 或者通过 API 调用

# 测试搜索（应该返回空结果，因为没有数据）
openclaw tool call memory_search --query "test"

# 期望输出：
# {
#   "results": [],
#   "provider": "openai",
#   "model": "text-embedding-3-small",
#   "hybrid": true
# }
```

### 7.2 测试 memory_get 工具
```bash
# 测试读取文件
openclaw tool call memory_get --path "MEMORY.md"

# 期望输出：
# {
#   "text": "...",
#   "path": "MEMORY.md"
# }
```

---

## 故障排查

### 问题 1：插件未加载
```bash
# 检查插件是否安装
openclaw plugins list

# 检查配置文件
cat ~/.openclaw/config.json | grep -A 10 "memory-milvus"

# 检查日志
tail -100 ~/.openclaw/logs/openclaw.log | grep -i "error\|plugin"
```

### 问题 2：Milvus 连接失败
```bash
# 检查 Milvus 地址是否正确
echo $MILVUS_ADDRESS

# 测试 Milvus 连接
python3 << 'EOF'
from pymilvus import connections
try:
    connections.connect("default", host="your-milvus-host", port="19530")
    print("Connected successfully!")
except Exception as e:
    print(f"Connection failed: {e}")
EOF
```

### 问题 3：嵌入提供商错误
```bash
# 检查 API Key
echo $OPENAI_API_KEY

# 测试 OpenAI 连接
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### 问题 4：Collection 创建失败
```bash
# 检查 OpenClaw 日志
tail -100 ~/.openclaw/logs/openclaw.log | grep -i "milvus\|collection"

# 手动创建 Collection 测试
python3 << 'EOF'
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

connections.connect("default", host="your-milvus-host", port="19530")

schema = CollectionSchema([
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
])

collection = Collection("test_collection", schema)
print("Collection created successfully!")
EOF
```

---

## 部署检查清单

- [ ] 插件已构建（`npm run build`）
- [ ] 插件已安装（`openclaw plugins install`）
- [ ] Milvus 地址已配置
- [ ] Milvus 连接已验证
- [ ] OpenClaw 配置已更新（`config.json`）
- [ ] OpenClaw 已重启
- [ ] 插件已加载（日志中看到 "memory-milvus: ready"）
- [ ] Collection 已创建（`openclaw-memory-{agentId}`）
- [ ] CLI 命令可用（`openclaw memory status`）
- [ ] 工具可用（`memory_search`, `memory_get`）
