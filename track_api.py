# -*- coding: utf-8 -*-
"""
用户追踪后端 API
采集 IP / User-Agent / 屏幕信息 / 语言 / 时区
设备指纹 device_id = UA + 屏幕 + tz + lang + IP前三段（hash）
生产/代理场景读取 X-Forwarded-For
"""

from fastapi import FastAPI, Request
from pydantic import BaseModel
from pymongo import MongoClient
from hashlib import sha256
from datetime import datetime
import os

# 可选：MaxMind 地理数据库
GEO_DB = os.getenv("GEOLITE_DB")
geo_reader = None
if GEO_DB and os.path.exists(GEO_DB):
    try:
        import geoip2.database
        geo_reader = geoip2.database.Reader(GEO_DB)
        print(f"✅ Loaded GeoIP database: {GEO_DB}")
    except Exception as e:
        print(f"⚠️ Failed to load GeoIP database: {e}")
        geo_reader = None

# MongoDB 配置
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "chatbot")
db = MongoClient(MONGO_URI)[MONGO_DB]

app = FastAPI(title="ChatBOT.EDU Tracking API", version="1.0.0")

class ClientPing(BaseModel):
    user_agent: str
    screen: dict | None = None  # {"w":..., "h":..., "dpr":...}
    tz: str | None = None
    lang: str | None = None
    cookie_uid: str | None = None

def get_ip(req: Request) -> str:
    """获取真实IP地址，支持代理场景"""
    xff = req.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return req.client.host

def geo_lookup(ip: str) -> dict | None:
    """IP地址地理信息查询"""
    if not geo_reader:
        return None
    try:
        r = geo_reader.city(ip)
        return {
            "country": r.country.iso_code,
            "region": r.subdivisions.most_specific.name,
            "city": r.city.name,
            "lat": r.location.latitude,
            "lon": r.location.longitude,
        }
    except Exception:
        return None

@app.post("/track")
async def track(p: ClientPing, request: Request):
    """用户设备信息追踪"""
    ip = get_ip(request)
    geo = geo_lookup(ip)
    
    # 设备指纹：UA + 屏幕 + 时区 + 语言 + IP前三段
    fp_src = f"{p.user_agent}|{p.screen}|{p.tz}|{p.lang}|{'.'.join(ip.split('.')[:3])}"
    device_id = "dev_" + sha256(fp_src.encode()).hexdigest()[:24]
    
    # 用户ID：优先使用cookie中的uid，否则基于设备指纹+IP生成
    uid = p.cookie_uid or "usr_" + sha256((device_id + ip).encode()).hexdigest()[:24]
    now = datetime.utcnow()

    # 创建或更新用户记录
    db.users.update_one(
        {"_id": uid},
        {
            "$setOnInsert": {"_id": uid, "created_at": now},
            "$set": {"last_seen_at": now}
        },
        upsert=True
    )
    
    # 更新或添加设备记录
    db.users.update_one(
        {"_id": uid, "devices.device_id": device_id},
        {"$set": {"devices.$.last_seen": now, "devices.$.ip": ip, "devices.$.geo": geo}}
    )
    
    # 如果设备不存在，添加新设备
    if db.users.count_documents({"_id": uid, "devices.device_id": device_id}) == 0:
        db.users.update_one(
            {"_id": uid},
            {"$push": {"devices": {
                "device_id": device_id,
                "user_agent": p.user_agent,
                "ip": ip,
                "geo": geo,
                "first_seen": now,
                "last_seen": now,
                "screen": p.screen,
                "tz": p.tz,
                "lang": p.lang
            }}}
        )

    return {"user_id": uid, "device_id": device_id}

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/stats")
async def stats():
    """统计信息"""
    user_count = db.users.count_documents({})
    conversation_count = db.conversations.count_documents({})
    message_count = db.messages.count_documents({})
    
    return {
        "users": user_count,
        "conversations": conversation_count,
        "messages": message_count,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8666)
